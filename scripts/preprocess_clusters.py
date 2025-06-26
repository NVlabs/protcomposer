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
import argparse
import pickle
from sklearn.cluster import SpectralClustering

parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--worker_id", type=int, default=0)
parser.add_argument("--res_per_cluster", type=int, default=20)
parser.add_argument("--dssp_weight", type=float, default=3)
parser.add_argument("--seq_weight", type=float, default=0.1)
parser.add_argument(
    "--pkl_dir", type=str, default="data/multiflow/train_set/processed_pdb"
)
parser.add_argument(
    "--target_dir", type=str, default="data/multiflow/train_set/processed_pdb_clusters"
)
parser.add_argument("--flat_dir", action="store_true")


args = parser.parse_args()

import math
import os
import tqdm
import numpy as np
import pydssp


if args.flat_dir:
    files = os.listdir(args.pkl_dir)
else:
    paths = os.listdir(args.pkl_dir)
    files = []
    for path in paths:
        files.extend(os.listdir(f"{args.pkl_dir}/{path}"))
print(len(files))


def do_job(file):
    if args.flat_dir:
        file_path = f"{args.pkl_dir}/{file}"
    else:
        file_path = f"{args.pkl_dir}/{file[1:3]}/{file}"
    try:
        with open(file_path, "rb") as f:
            prot = pickle.load(f)
    except:
        print("Failure", file)
        return None

    mask = prot["bb_mask"].astype(bool)
    atom37 = prot["atom_positions"][mask]

    bb_pos = np.concatenate([atom37[:, :3, :], atom37[:, 4:5, :]], axis=1)  # (L, 4, 3)

    try:
        dssp = pydssp.assign(
            bb_pos, out_type="index"
        )  # 0: loop,  1: alpha-helix,  2: beta-strand
    except:
        print("Failure", file, bb_pos.shape)
        return None

    pos = atom37[:, 1, :]
    n_clusters = math.ceil(len(pos) / args.res_per_cluster)

    dssp_one_hot = dssp[:, None] == np.arange(3)
    feat = np.concatenate(
        [
            pos,
            dssp_one_hot * args.dssp_weight,
            np.arange(len(dssp))[:, None] * args.seq_weight,
        ],
        axis=1,
    )

    distmat = np.square(feat[None] - feat[:, None]).sum(-1) ** 0.5
    r = 2
    W = -(distmat**2) / (2 * r**2)
    W = np.exp(W)
    n_clusters = math.ceil(len(dssp) / args.res_per_cluster)
    labels = (
        SpectralClustering(n_clusters=n_clusters, affinity="precomputed").fit(W).labels_
    )

    dssp_count = np.zeros((n_clusters, 3))
    np.add.at(dssp_count, labels, dssp_one_hot)

    centers = np.zeros((n_clusters, 3))
    np.mean.at(centers, labels, pos)

    clustering = {
        "centers": centers,
        "labels": labels,
        "dssp_count": dssp_count,
    }

    return clustering


out = {}
for job in tqdm.tqdm(files):
    out[job] = do_job(job)

with open(args.target_dir, "wb") as f:
    f.write(pickle.dumps(out))
