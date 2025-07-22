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

import os, torch
from biotite.sequence.io import fasta
import esm
import numpy as np
from openfold.np import protein, residue_constants
from proteinblobs.designability_utils import get_aligned_rmsd
from proteinblobs.utils import atom37_to_pdb
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True)
parser.add_argument("--filter_missing_res", action="store_true", default=False)
parser.add_argument("--ca_only", action="store_true", default=False)
args = parser.parse_args()

tmpdir = f"/tmp/{os.getpid()}"
os.makedirs(tmpdir, exist_ok=True)

print("Loading ESMFold model for designability evaluation")
esmf_model = esm.pretrained.esmfold_v1().eval()
esmf_model = esmf_model.to("cuda")


def designability(path):
    sample_path = os.path.join(tmpdir, f"sample.pdb")
    if args.filter_missing_res:
        prot = protein.from_pdb_string(open(path, "r").read())
        ca_idx = residue_constants.atom_order["CA"]
        c_idx = residue_constants.atom_order["C"]
        n_idx = residue_constants.atom_order["N"]
        present_mask = prot.atom_mask[:, [ca_idx, c_idx, n_idx]].all(-1)
        pos = prot.atom_positions[present_mask]
        atom37_to_pdb(pos[None], sample_path)
    else:
        os.system(f"cp {path} {tmpdir}/sample.pdb")

    seq_numeric = np.load(path.replace(".pdb", "_sequence.npy"))
    seq = "".join([residue_constants.restypes_with_x[r] for r in seq_numeric])
    seq = seq.replace("X", "A")

    with torch.no_grad():
        output = esmf_model.infer(seq)

    with open(sample_path) as f:
        prot = protein.from_pdb_string(f.read())
    out_ca_pos = output["positions"][-1].squeeze()[:, 2].cpu().numpy()
    rmsd = get_aligned_rmsd(prot.atom_positions[:, 1], out_ca_pos)

    return {"rmsd": [rmsd]}


files = os.listdir(args.dir)
files = [f for f in files if ".pdb" in f]
out = {}
for f in files:
    out[f] = designability(f"{args.dir}/{f}")
with open(f"{args.dir}/designability_codesign.json", "w") as f:
    f.write(json.dumps(out, indent=4))
