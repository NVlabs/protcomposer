# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import subprocess

import esm, tqdm, torch
import os
from biotite.sequence.io import fasta
import numpy as np

from proteinblobs.utils import atom37_to_pdb
from openfold.np import residue_constants



def run_designability(
    prots, pmpnn_path, process_rank=0, seqs_per_struct=8, sequences=None
):
    # prots in list of np arrays of ( L, 3)

    print("Loading ESMFold model for designability evaluation")
    torch.cuda.empty_cache()
    esmf_model = esm.pretrained.esmfold_v1().eval()
    esmf_model = esmf_model.to("cuda")

    all_tm_scores = []
    all_rmsds = []
    top_rmsds = []
    top_tm_scores = []
    for i, prot in tqdm.tqdm(enumerate(prots), desc="Running PMPNN and ESMFold"):
        sample_dir = path = os.path.join(
            os.environ["MODEL_DIR"],
            f"tmp_design_dir_process{process_rank}",
            f"sample{i}_process{process_rank}",
        )
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, f"sample.pdb")

        rmsds = []
        #tm_scores = []
        if sequences is not None:
            restypes = residue_constants.restypes + ["X"]
            seqs = ["".join(map(lambda x: restypes[x], sequences[i]))]

        else:
            atom37_to_pdb(prot[None], sample_path)
            run_pmpnn(
                pdb_dir=sample_dir, num_seqs=seqs_per_struct, pmpnn_path=pmpnn_path
            )
            mpnn_fasta_path = os.path.join(
                sample_dir, "seqs", os.path.basename(sample_path).replace(".pdb", ".fa")
            )
            fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)

            seqs = [v for k, v in fasta_seqs.items()]
            seqs = seqs[
                1:
            ]  # remove the first sequence, which is the input sequence (and AAAAAAA... if there is no input sequence to the structure)

        for j, seq in enumerate(seqs):
            seq = seq.replace("X", "A")

            with torch.no_grad():
                output = esmf_model.infer(seq)

            out_ca_pos = output["positions"][-1].squeeze()[:, 2].cpu().numpy()
            #_, tm_score = get_tm_score(prot[:, 1], out_ca_pos, seq, seq)
            rmsd = get_aligned_rmsd(prot[:, 1], out_ca_pos)
            rmsds.append(rmsd)
            #tm_scores.append(tm_score)
        all_rmsds.append(np.array(rmsds).mean())
        #all_tm_scores.append(np.array(tm_scores).mean())
        top_rmsds.append(np.array(rmsds).min())
        #top_tm_scores.append(np.array(tm_scores).max())
    del esmf_model
    torch.cuda.empty_cache()
    #all_tm_scores = np.array(all_tm_scores)
    all_rmsds = np.array(all_rmsds)
    top_rmsds = np.array(top_rmsds)
    #top_tm_scores = np.array(top_tm_scores)

    return {
        #"tm_score": all_tm_scores,
        "rmsd": all_rmsds,
        #"top_tm_score": top_tm_scores,
        "top_rmsd": top_rmsds,
    }


def run_pmpnn(pdb_dir, num_seqs=8, pmpnn_path="../ProteinMPNN", ca_only=False):
    os.makedirs(os.path.join(pdb_dir, "seqs"), exist_ok=True)
    parsed_chains_path = os.path.join(pdb_dir, "seqs", "parsed_chains.jsonl")

    process = subprocess.Popen(
        [
            "python",
            os.path.join(pmpnn_path, "helper_scripts/parse_multiple_chains.py"),
            f"--input_path={pdb_dir}",
            f"--output_path={parsed_chains_path}",
        ]
    )
    _ = process.wait()

    pmpnn_args = [
        "python",
        os.path.join(pmpnn_path, "protein_mpnn_run.py"),
        "--out_folder",
        pdb_dir,
        "--jsonl_path",
        parsed_chains_path,
        "--num_seq_per_target",
        str(num_seqs),
        "--sampling_temp",
        "0.1",
        "--seed",
        "38",
        "--batch_size",
        "1",
    ]
    if ca_only:
        pmpnn_args.append("--ca_only")
    print(" ".join(pmpnn_args))

    process = subprocess.run(pmpnn_args)


def get_aligned_rmsd(pos_1, pos_2):
    aligned_pos_1 = rigid_transform_3D(pos_1, pos_2)[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))


def get_tm_score(pos_1, pos_2, seq_1, seq_2):
    # tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2


def rigid_transform_3D(A, B, verbose=False):
    # Transforms A to look like B
    # https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected
