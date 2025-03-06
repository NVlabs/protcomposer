# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from proteinblobs import blobs as pb
import esm
from openfold.np import protein
import tqdm
import argparse
import pydssp
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True)
args = parser.parse_args()

files = os.listdir(args.dir)
files = sorted([f for f in files if ".pdb" in f])

df = []
for i, pdb in tqdm.tqdm(enumerate(files)):
    prot = protein.from_pdb_string(open(f"{args.dir}/{pdb}").read())
    blobs = dict(np.load(f"{args.dir}/{pdb.replace('pdb', 'npz')}"))

    pos = np.concatenate(
        [prot.atom_positions[:, :3, :], prot.atom_positions[:, 4:5, :]], axis=1
    )  # (L, 4, 3)
    dssp = pydssp.assign(pos, out_type="index")
    pos = pos[:, 1]
    blobs = [
        {
            "pos": blobs["pos"][i],
            "covar": blobs["covar"][i],
            "dssp": blobs["dssp"][i],
            "count": blobs["counts"][i],
        }
        for i in range(len(blobs["dssp"]))
    ]
    df.append(
        {
            "name": pdb,
            "coverage": pb.blob_coverage(pos, dssp, blobs, structured_only=True),
            "misplacement": pb.blob_misplacement(
                pos, dssp, blobs, structured_only=True
            ),
            "accuracy": pb.blob_accuracy(pos, dssp, blobs, structured_only=True),
            "likelihood": pb.blob_likelihood(pos, dssp, blobs, structured_only=True),
            "soft_accuracy": pb.soft_blob_accuracy(
                pos, dssp, blobs, structured_only=True
            ),
            "reblob_jsd": pb.reblob_jsd(pos, dssp, blobs, use_dssp=True),
            "shannon_6.5": pb.shannon_complexity(pos, dssp, 6.5),
        }
    )
df = pd.DataFrame(df)
df.set_index("name").to_csv(f"{args.dir}/alignment.csv")
print(df)
for colname in df.columns.tolist():
    if colname != "name":
        print(f"{colname}: {df[colname].mean()}")
