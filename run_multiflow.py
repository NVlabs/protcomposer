# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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



