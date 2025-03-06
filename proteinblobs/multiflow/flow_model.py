# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch import nn

from .node_feature_net import NodeFeatureNet
from .edge_feature_net import EdgeFeatureNet
from .cross_attention import InvariantCrossAttention
from . import ipa_pytorch
from .data import utils as du
import math

def positional_encoding(x, c, max_period=10000, min_period=None):
    min_period = min_period or max_period / 10000 # second one recommended
    freqs = torch.exp(-torch.linspace(math.log(min_period), math.log(max_period), c // 2, device=x.device))
    emb = freqs * x.unsqueeze(-1) # [..., C]
    return torch.cat([torch.sin(emb), torch.cos(emb)], -1) # [..., 2C]

class FlowModel(nn.Module):

    def __init__(self, model_conf, args):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self.args = args
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)
        if self.args.use_latents:
            self.latent_embedder = ipa_pytorch.Linear(32, model_conf.node_features.c_s, init="final")

        if self.args.blob_attention:
            self.ground_feature_net = nn.Embedding(3, self._ipa_conf.c_s)
            self.ground_trace_feature_net = nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s)
            self.ground_size_feature_net = nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s)

        if self._model_conf.aatype_pred:
            node_embed_size = self._model_conf.node_embed_size
            self.aatype_pred_net = nn.Sequential(
                nn.Linear(node_embed_size, node_embed_size),
                nn.ReLU(),
                nn.Linear(node_embed_size, node_embed_size),
                nn.ReLU(),
                nn.Linear(node_embed_size, self._model_conf.aatype_pred_num_tokens),
            )

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            if self.args.blob_attention:
                self.trunk[f'ica_{b}'] = InvariantCrossAttention(self._ipa_conf)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=self._model_conf.transformer_dropout,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = nn.TransformerEncoder(tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(tfmr_in, self._ipa_conf.c_s, init="final")
            
            if self.args.extra_attn_layer:
                self.trunk[f'cond_tfmr_{b}'] = nn.TransformerEncoder(tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
                self.trunk[f'post_cond_tfmr_{b}'] = ipa_pytorch.Linear(tfmr_in, self._ipa_conf.c_s, init="final")
                
            
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        if args.freeze_weights:
            for name, p in self.named_parameters():
                if ('ica' not in name) and ('cond_tfmr' not in name) and ('ground' not in name):
                    p.requires_grad_(False)

    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        chain_index = input_feats['chain_idx']
        res_index = input_feats['res_idx']
        latents = input_feats['latents']
        so3_t = input_feats['so3_t']
        r3_t = input_feats['r3_t']
        cat_t = input_feats['cat_t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        aatypes_t = input_feats['aatypes_t'].long()
        trans_sc = input_feats['trans_sc']
        aatypes_sc = input_feats['aatypes_sc']

        ##########
        if self.args.blob_attention:
            grounding_feats = input_feats['grounding_feat']
            grounding_pos = input_feats['grounding_pos']
            grounding_mask = input_feats['grounding_mask']

            ###
            B, L, _ = grounding_pos.shape
            grounding_covar = grounding_feats[:,:,-9:].view(B, L, 3, 3)
            trace = grounding_covar[:,:,torch.arange(3),torch.arange(3)].sum(-1)
            grounding_covar = grounding_covar / (trace[:,:,None,None] + 1e-3) # "normalized covariance"
            ###
            grounding_feats = (
                self.ground_feature_net(grounding_feats[...,0].long()) + 
                self.ground_size_feature_net(
                    positional_encoding(grounding_feats[...,1].float(), self._ipa_conf.c_s)
                ) +
                self.ground_trace_feature_net(
                    positional_encoding(trace, self._ipa_conf.c_s)
                )
            )
            assert not torch.isnan(grounding_feats).any()
        ##########
        
        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t=so3_t,
            r3_t=r3_t,
            cat_t=cat_t,
            res_mask=node_mask,
            diffuse_mask=diffuse_mask,
            chain_index=chain_index,
            pos=res_index,
            aatypes=aatypes_t,
            aatypes_sc=aatypes_sc,
        )
        if self.args.use_latents:
            init_node_embed = init_node_embed + self.latent_embedder(latents)[:,None,:]

        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,
            trans_sc,
            edge_mask,
            diffuse_mask,
            chain_index
        )
        

        # Initial rigids
        init_rigids = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            
            ############
            if self.args.blob_attention:
                ica_out = self.trunk[f'ica_{b}'](
                    node_embed, 
                    curr_rigids, 
                    grounding_pos * du.ANG_TO_NM_SCALE,  # FIX, 
                    grounding_feats, 
                    grounding_covar,
                    grounding_mask
                )
                node_embed = node_embed + ica_out * (1 if self.training else self.args.inference_gating)
            ###########

            def heterogenous_attention(layer, post_layer, node_embed, node_mask, grounding_feats, grounding_mask):
            
                node_embed_ = torch.cat([node_embed, grounding_feats], 1)
                node_mask_ = torch.cat([node_mask, grounding_mask], 1)
                cond_tfmr_out = layer(node_embed_, src_key_padding_mask=(1 - node_mask_).bool())
                L = node_embed.shape[1]

                cond_tfmr_out = post_layer(cond_tfmr_out)
                node_embed = node_embed + cond_tfmr_out[:,:L] * (1 if self.training else self.args.inference_gating)
                grounding_feats = grounding_feats + cond_tfmr_out[:,L:]

                return node_embed, grounding_feats
                        
            if self.args.blob_attention and not self.args.extra_attn_layer:
                node_embed, grounding_feats = heterogenous_attention(
                    self.trunk[f'seq_tfmr_{b}'],
                    self.trunk[f'post_tfmr_{b}'],
                    node_embed, node_mask, grounding_feats, grounding_mask)

            else:
                seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                    node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
                node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
                    
            ################
            if self.args.blob_attention and self.args.extra_attn_layer:
                node_embed, grounding_feats = heterogenous_attention(
                        self.trunk[f'cond_tfmr_{b}'],
                        self.trunk[f'post_cond_tfmr_{b}'],
                        node_embed, node_mask, grounding_feats, grounding_mask)

            ################
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None])
            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        if self._model_conf.aatype_pred:
            pred_logits = self.aatype_pred_net(node_embed)
            pred_aatypes = torch.argmax(pred_logits, dim=-1)
            if self._model_conf.aatype_pred_num_tokens == du.NUM_TOKENS + 1:
                pred_logits_wo_mask = pred_logits.clone()
                pred_logits_wo_mask[:, :, du.MASK_TOKEN_INDEX] = -1e9
                pred_aatypes = torch.argmax(pred_logits_wo_mask, dim=-1)
            else:
                pred_aatypes = torch.argmax(pred_logits, dim=-1)
        else:
            pred_aatypes = aatypes_t
            pred_logits = nn.functional.one_hot(
                pred_aatypes, num_classes=self._model_conf.aatype_pred_num_tokens
            ).float()
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'pred_logits': pred_logits,
            'pred_aatypes': pred_aatypes,
        }
