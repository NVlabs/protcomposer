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

import os
import tqdm
import argparse
import numpy as np

import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True)
parser.add_argument('--outdir', type=str, default=None)
args = parser.parse_args()




if args.outdir is None:
    args.outdir = args.dir


def calculate_secondary_structure(pdb_file):
    """
    Calculate secondary structures using Biotite.
    """
    file = pdb.PDBFile.read(pdb_file)
    array = file.get_structure()
    sse = struc.annotate_sse(array[0])

    helices = (sse == 'a').sum()  # Helices
    strands = (sse == 'b').sum()  # Strands
    coils = (sse == 'c').sum()    # Coils
    total = len(sse)

    helix_percentage = helices / total if total > 0 else 0
    strand_percentage = strands / total if total > 0 else 0
    coil_percentage = coils / total if total > 0 else 0

    return helix_percentage, strand_percentage, coil_percentage

def preprocess_pdb_files(pdb_files):
    """
    Ensure every PDB file listed in the specified text file has valid occupancy and B-factor values otherwise will be filled with a default value
    """
    for pdb_path in pdb_files:
        pdb_path = str(pdb_path)
        corrected_content = []
        with open(pdb_path, 'r') as file:
            for line in file:
                if line.startswith(("ATOM", "HETATM")):
                    # Occupancy is located from columns 55 to 60 In Case an Error is thrown
                    occupancy_value = line[54:60].strip()
                    if not occupancy_value:
                        line = line[:54] + '1.00' + line[60:]  # Insert default occupancy In Case an Error is thrown

                    # B-factor is located from columns 61 to 66 In Case an Error is thrown
                    b_factor_value = line[60:66].strip()
                    if not b_factor_value:
                        line = line[:60] + '0.00' + line[66:]  # Insert default B-factor In Case an Error is thrown

                corrected_content.append(line)

        with open(pdb_path, 'w') as file:
            file.writelines(corrected_content)

files = os.listdir(args.dir)
files = sorted([f for f in files if '.pdb' in f])

helicity = []
for i, file in tqdm.tqdm(enumerate(files)):
    preprocess_pdb_files([f"{args.dir}/{file}"])
    helix_percent, _, _ = calculate_secondary_structure(f"{args.dir}/{file}")
    helicity.append(helix_percent)
old_dict = dict(np.load(f"{args.outdir}/res.npz"))
old_dict['helicity'] = np.array(helicity)
np.savez(f"{args.outdir}/res.npz", **old_dict)


