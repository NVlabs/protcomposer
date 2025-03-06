# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from proteinblobs.parsing import parse_train_args

args = parse_train_args()

import torch
from omegaconf import OmegaConf
import numpy as np
from proteinblobs.multiflow.flow_model import FlowModel
from proteinblobs.multiflow.data.interpolant import Interpolant
from proteinblobs.multiflow.data import utils as du
from proteinblobs.utils import atom37_to_pdb
from proteinblobs.dataset import seq_collate, GenieDBDataset

import copy
from tqdm import tqdm

cfg = OmegaConf.load("multiflow_config.yaml")
cfg.model.edge_features.self_condition = (
    args.self_condition
)  # only the cfg.interpolant.self_condition is used anywhere in the code
cfg.interpolant.self_condition = args.self_condition

print("init model")
args_uncond = copy.deepcopy(args)
args_uncond.freeze_weights = True
args_uncond.extra_attn_layer = False
args_uncond.blob_attention = False
model_uncond = FlowModel(cfg.model, args_uncond)
ckpt = torch.load(args.pretrained_mf_path, map_location="cuda")
model_uncond.load_state_dict(
    {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
)
model_uncond = model_uncond.to("cuda")

print("init dataset")
dataset = GenieDBDataset(args)
dataset[0]
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=seq_collate,
    shuffle=True,
)

interpolant = Interpolant(cfg.interpolant, args)
interpolant.set_device("cuda")
for batch in tqdm(loader):
    true_bb_pos = None
    num_batch, sample_length = batch["mask"].shape
    prot_traj, model_traj = interpolant.sample(
        batch["mask"].cuda(),
        model_uncond,
        num_timesteps=args.num_timesteps,
        separate_t=cfg.inference.interpolant.codesign_separate_t,
    )
    diffuse_mask = torch.ones(1, sample_length)
    atom37_traj = [x[0] for x in prot_traj]
    atom37_model_traj = [x[0] for x in model_traj]

    bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
    noisy_traj_length = bb_trajs.shape[1]
    assert bb_trajs.shape == (num_batch, noisy_traj_length, sample_length, 37, 3)

    model_trajs = du.to_numpy(torch.stack(atom37_model_traj, dim=0).transpose(0, 1))
    clean_traj_length = model_trajs.shape[1]
    assert model_trajs.shape == (num_batch, clean_traj_length, sample_length, 37, 3)

    aa_traj = [x[1] for x in prot_traj]
    clean_aa_traj = [x[1] for x in model_traj]

    aa_trajs = du.to_numpy(torch.stack(aa_traj, dim=0).transpose(0, 1).long())
    assert aa_trajs.shape == (num_batch, noisy_traj_length, sample_length)
    for i in range(aa_trajs.shape[0]):
        for j in range(aa_trajs.shape[2]):
            if aa_trajs[i, -1, j] == du.MASK_TOKEN_INDEX:
                print("WARNING mask in predicted AA")
                aa_trajs[i, -1, j] = 0
    clean_aa_trajs = du.to_numpy(
        torch.stack(clean_aa_traj, dim=0).transpose(0, 1).long()
    )
    assert clean_aa_trajs.shape == (num_batch, clean_traj_length, sample_length)

    samples = bb_trajs[:, -1]
    for i, sample in enumerate(samples):
        sample_path = f"/lustre/fsw/portfolios/nvr/projects/nvr_lpr_compgenai/hstaerk_bjing/multiflow_output/{batch['name'][i].replace('.pdb', '')}_{np.random.randint(100000)}.pdb"
        atom37_to_pdb(sample[None], sample_path)
