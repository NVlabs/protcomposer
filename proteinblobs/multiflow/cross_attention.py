# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch.nn as nn

import torch
import math

def positional_encoding(x, channels, max_period=10000, min_period=None):
    min_period = min_period or max_period / 10000 # second one recommended
    freqs = torch.exp(-torch.linspace(math.log(min_period), math.log(max_period), channels // 2, device=x.device))
    emb = freqs * x.unsqueeze(-1) # [..., C]
    return torch.cat([torch.sin(emb), torch.cos(emb)], -1) # [..., 2C]



class InvariantCrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pos_emb_dim = 64
        self.linear_pos = nn.Linear(3 * pos_emb_dim, cfg.c_s)
        self.linear_kv = nn.Linear(cfg.c_s, 2 * cfg.c_s)
        self.linear_covar = nn.Linear(9, cfg.c_s)
        
        self.linear_q = nn.Linear(cfg.c_s, cfg.c_s)
        self.linear_out = nn.Linear(cfg.c_s, cfg.c_s)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.beta.requires_grad_(False)
        self.inf = 1e10

    def forward(self, s, r, x, y, c, mask):
        '''
        s: frame features
        r: frames
        x: point positions
        y: point features
        c: blob covariance
        mask: point mask
        '''
        
        # [B, L, N, 3]
        relpos = r[:,:,None].invert_apply(x[:,None])

        # [B, L, 3, 3]
        R = r._rots.get_rot_mats()

        # [B, L, N, 3, 3]  
        covar = torch.einsum('blij,bnjk,blkm->blnim', R.transpose(-1, -2), c, R)

        # [B, L, N, C]
        relpos = positional_encoding(relpos, 64, max_period=1000) # 1000 nm???
        B, L, N, _, _ = relpos.shape

        # [B, L, N, C]
        y = y[:,None] + self.linear_pos(relpos.view(B, L, N, 3*64)) + self.linear_covar(covar.view(B, L, N, 9))
        B, L, N, C = y.shape
        
        # [B, L, N, C]
        k, v = self.linear_kv(y).chunk(2, dim=-1)

        # [B, L, C]
        q = self.linear_q(s)

        H = self.cfg.no_heads
        k = k.view(B, L, N, H, C//H)
        v = v.view(B, L, N, H, C//H)
        q = q.view(B, L, 1, H, C//H)

        # [B, L, N, H]
        attn = torch.einsum('BLNHC,BLNHC->BLNH', q, k) / math.sqrt(C//H)
        attn = attn + (mask[:,None,:,None] - 1) * self.inf
        attn = torch.softmax(attn, dim=-2)

        # [B, L, C]
        output = torch.einsum('BLNH,BLNHC->BLHC', attn, v).reshape(B, L, C)
        output = self.linear_out(output)
        
        output = self.beta * torch.tanh(self.gamma) * output

        return output
        

        



        
        