# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from omegaconf import OmegaConf
import os, torch
import pytorch_lightning as pl
import numpy as np
from proteinblobs.multiflow_wrapper import MultiflowWrapper
from openfold.np import protein
from proteinblobs.utils import upgrade_state_dict, create_full_prot
import argparse

args = argparse.Namespace()
args.__dict__.update({
    'multiflow_yaml': "../multiflow/weights/config.yaml"
})
model = MultiflowWrapper(args)
ckpt_path = "../multiflow/weights/last.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(ckpt['state_dict'])
model.eval().cuda()

out, _ = model.inference(num_batch=2, sample_length=128)

for i, prot in enumerate(out):
    prot = create_full_prot(prot)
    
    with open(f'test{i}.pdb', 'w') as f:
        f.write(protein.to_pdb(prot))



