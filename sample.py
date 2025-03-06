# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='workdir/tune_SC_elli_warmup/epoch=12-step=59683.ckpt')
parser.add_argument('--num_blobs', type=int, default=5)
parser.add_argument('--num_prots', type=int, default=5)
parser.add_argument('--nu', type=int, default=10) # vary the blob anisotropy [10, 20, 50, 100]
parser.add_argument('--psi', type=int, default=5) # FIX this to give proteins with O(200) residues
parser.add_argument('--sigma', type=int, default=8) # vary this compactness parameter
parser.add_argument('--helix_frac', type=float, default=0.5)
parser.add_argument('--num_residues', type=int, default=160)
parser.add_argument('--seed', type=int, default=137)
parser.add_argument('--num_timesteps', type=int, default=500)
parser.add_argument('--inference_rot_scaling', type=float, default=10)
parser.add_argument('--multiflow', action='store_true')
parser.add_argument('--outdir', default='outpdb/default')
parser.add_argument('--guidance', type=float, default=None)
args = parser.parse_args()

from pymol import cmd
import torch
from proteinblobs.parsing import parse_train_args
from proteinblobs.multiflow_wrapper import MultiflowWrapper
from proteinblobs import blobs
from proteinblobs.utils import atom37_to_pdb
import os, contextlib
import numpy as np
import tqdm

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch_state = torch.seed()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
        torch.manual_seed(seed)
        

np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(args.outdir, exist_ok=True)
blobss = []
for _ in tqdm.tqdm(range(args.num_prots), desc='sampling dataset'):
    pos, covar = blobs.sample_blobs(args.num_blobs, nu=args.nu, psi=(1/args.nu)*args.psi**2*np.eye(3), sigma=args.sigma)
    is_helix = np.random.rand(args.num_blobs) < args.helix_frac
    volume = np.linalg.det(covar)**0.5
    
    counts = np.where(
        is_helix,
        blobs.alpha_slope * volume + blobs.alpha_intercept,
        blobs.beta_slope * volume + blobs.beta_intercept,
    ).astype(int)
    
    dssp = np.where(is_helix, 1, 2)
    blobss.append((pos, covar, counts, dssp))
    

if args.multiflow:
    
    args_ = argparse.Namespace()
    args_.__dict__.update({
        'finetune': True,
        'multiflow_yaml': "multiflow_config.yaml",
        'num_timesteps': args.num_timesteps,
        'self_condition': True,
        'blob_attention': False,
        'extra_attn_layer': False,
        'freeze_weights': True,
        'pretrained_mf_path': 'weights/last.ckpt',
        'inference_rot_scaling': args.inference_rot_scaling,
    })

    model = MultiflowWrapper(args_).eval().cuda()
else:    
    ckpt = torch.load(args.ckpt)
    
    args_ = ckpt['hyper_parameters']['args']
    if args.guidance is not None: #--guidance --sc_guidance_mode separate --interpolate_sc --interpolate_sc_weight 0.8 --guidance_weight 0.8 --seq_guidance_weight 0.8
        args_.__dict__.update({
            'guidance': True,
            'sc_guidance_mode': 'separate',
            'num_timesteps': args.num_timesteps,
            'interpolate_sc': True,
            'pretrained_mf_path': 'weights/last.ckpt',
            'interpolate_sc_weight': args.guidance,
            'guidance_weight': args.guidance,
            'seq_guidance_weight': args.guidance,
        })
    model = MultiflowWrapper(args_)
    model.load_state_dict(ckpt['state_dict'])
    model = model.eval().cuda()
    
    if args.guidance is not None:
        with temp_seed(args.seed):
            model.on_validation_epoch_start()

for i, (pos, covar, counts, dssp) in tqdm.tqdm(enumerate(blobss), desc='running inference'):

    batch = {}
    if args.multiflow:
        batch['res_mask'] = np.ones((1, args.num_residues), dtype=np.float32)
        batch['grounding_pos'] = batch['grounding_feat'] = batch['grounding_mask'] = np.zeros(1)
    else:
        batch['res_mask'] = np.ones((1, counts.sum()), dtype=np.float32)
        batch['grounding_pos'] = pos[None].astype(np.float32)
        batch['grounding_feat'] = np.zeros((1, args.num_blobs, 11), dtype=np.float32)
        batch['grounding_feat'][:,:,0] = dssp # all helix for now
        batch['grounding_feat'][:,:,1] = counts
        batch['grounding_feat'][:,:,2:] = covar.reshape(1, args.num_blobs, 9)
        batch['grounding_mask'] = np.ones((1, args.num_blobs), dtype=np.float32)
    
        
    batch = {k: torch.from_numpy(v).cuda() for k, v in batch.items()}
    batch['latents'] = 0
    if not args.multiflow:
        np.savez(f'{args.outdir}/blobs_{i}.npz', pos=pos, covar=covar, counts=counts, dssp=dssp)
    with torch.no_grad():
        samples, sequences = model.inference(batch)
    atom37_to_pdb(samples, f'{args.outdir}/blobs_{i}.pdb')
    np.save(f'{args.outdir}/blobs_{i}_sequence.npy', sequences[0])
    