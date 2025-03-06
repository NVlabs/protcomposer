# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import argparse, tqdm
from openfold.np import protein
import openfold.np.residue_constants as rc
import pandas as pd
from multiprocessing import Pool
import pydssp

parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, default='../afdb')
parser.add_argument('--outdir', type=str, default='./afdb_npz')
parser.add_argument('--outcsv', type=str, default='./afdb.csv')
parser.add_argument('--num_workers', type=int, default=15)
args = parser.parse_args()


def main():
    pdbs = list(map(lambda x: x.strip(), open('index.txt')))
    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    infos = list(tqdm.tqdm(__map__(unpack_mmcif, pdbs), total=len(pdbs)))
    if args.num_workers > 1:
        p.__exit__(None, None, None)
    df = pd.DataFrame(infos).set_index('name')
    df.to_csv(args.outcsv)    
    
def unpack_mmcif(name):
    path = f"{args.indir}/{name}.pdb"

    with open(path, 'r') as f:
        prot = protein.from_pdb_string(f.read())

    atom37 = prot.atom_positions
    # take N, CA, C, and O. The order in atom37 is N, CA, C, CB, O ... (see atom_types in residue_constants.py)
    bb_pos = np.concatenate([atom37[ :, :3, :], atom37[ :, 4:5, :]], axis=1)  # (L, 4, 3)
    dssp = pydssp.assign(bb_pos, out_type='index') # 0: loop,  1: alpha-helix,  2: beta-strand

    data = {
        'atom_positions': prot.atom_positions,
        'aatype': prot.aatype,
        'atom_mask': prot.atom_mask,
        'b_factors': prot.b_factors,
        'dssp': dssp,
    }
    np.savez(f"{args.outdir}/{name}.npz", **data)

    out = {
        'name': name,
        'seqres': "".join([rc.restypes[x] for x in prot.aatype]),
        'length': len(prot.aatype),
        'plddt': 100 - prot.b_factors.mean(),
        'loop_frac': (dssp == 0).mean(),
        'helix_frac': (dssp == 1).mean(),
        'sheet_frac': (dssp == 2).mean(),
    }
    
    return out
    
if __name__ == "__main__":
    main()