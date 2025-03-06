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
import re
import subprocess

import numpy as np
import torch

from .multiflow.data.all_atom import compute_backbone
import openfold.np.protein as protein


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


def run_pmpnn(input_dir, output_path, pmpnn_path="../ProteinMPNN"):
    os.makedirs(os.path.join(input_dir, "seqs"), exist_ok=True)
    process = subprocess.Popen(
        [
            "python",
            os.path.join(pmpnn_path, "helper_scripts/parse_multiple_chains.py"),
            f"--input_path={input_dir}",
            f"--output_path={output_path}",
        ]
    )
    _ = process.wait()

    pmpnn_args = [
        "python",
        os.path.join(pmpnn_path, "protein_mpnn_run.py"),
        "--out_folder",
        input_dir,
        "--jsonl_path",
        output_path,
        "--num_seq_per_target",
        "8",
        "--sampling_temp",
        "0.1",
        "--seed",
        "38",
        "--batch_size",
        "1",
    ]
    print(" ".join(pmpnn_args))

    process = subprocess.run(pmpnn_args)


def get_aligned_rmsd(pos_1, pos_2):
    aligned_pos_1 = rigid_transform_3D(pos_1, pos_2)[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))


def upgrade_state_dict(state_dict, prefixes=["encoder.sentence_encoder.", "encoder."]):
    """Removes prefixes like 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def compute_distogram_loss(
    logits,
    pseudo_beta,
    pseudo_beta_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    **kwargs,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries**2

    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean


class HarmonicPrior:
    def __init__(self, N=256, a=3 / (3.8**2)):
        J = torch.zeros(N, N)
        for i, j in zip(np.arange(N - 1), np.arange(1, N)):
            J[i, i] += a
            J[j, j] += a
            J[i, j] = J[j, i] = -a
        D, P = torch.linalg.eigh(J)
        D_inv = 1 / D
        D_inv[0] = 0
        self.P, self.D_inv = P, D_inv
        self.N = N

    def to(self, device):
        self.P = self.P.to(device)
        self.D_inv = self.D_inv.to(device)
        return self

    def sample(self, batch_dims=()):
        return self.P @ (
            torch.sqrt(self.D_inv)[:, None]
            * torch.randn(*batch_dims, self.N, 3, device=self.P.device)
        )


def adjust_oxygen_pos(atom_37: torch.Tensor, pos_is_known=None) -> torch.Tensor:
    """
    Imputes the position of the oxygen atom on the backbone by using adjacent frame information.
    Specifically, we say that the oxygen atom is in the plane created by the Calpha and C from the
    current frame and the nitrogen of the next frame. The oxygen is then placed c_o_bond_length Angstrom
    away from the C in the current frame in the direction away from the Ca-C-N triangle.

    For cases where the next frame is not available, for example we are at the C-terminus or the
    next frame is not available in the data then we place the oxygen in the same plane as the
    N-Ca-C of the current frame and pointing in the same direction as the average of the
    Ca->C and Ca->N vectors.

    Args:
        atom_37 (torch.Tensor): (N, 37, 3) tensor of positions of the backbone atoms in atom_37 ordering
                                which is ['N', 'CA', 'C', 'CB', 'O', ...]
        pos_is_known (torch.Tensor): (N,) mask for known residues.
    """

    N = atom_37.shape[0]
    assert atom_37.shape == (N, 37, 3)

    # Get vectors to Carbonly from Carbon alpha and N of next residue. (N-1, 3)
    # Note that the (N,) ordering is from N-terminal to C-terminal.

    # Calpha to carbonyl both in the current frame.
    calpha_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[:-1, 1, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[:-1, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # For masked positions, they are all 0 and so we add 1e-7 to avoid division by 0.
    # The positions are in Angstroms and so are on the order ~1 so 1e-7 is an insignificant change.

    # Nitrogen of the next frame to carbonyl of the current frame.
    nitrogen_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[1:, 0, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[1:, 0, :], keepdim=True, dim=1) + 1e-7
    )

    carbonyl_to_oxygen: torch.Tensor = (
        calpha_to_carbonyl + nitrogen_to_carbonyl
    )  # (N-1, 3)
    carbonyl_to_oxygen = carbonyl_to_oxygen / (
        torch.norm(carbonyl_to_oxygen, dim=1, keepdim=True) + 1e-7
    )

    atom_37[:-1, 4, :] = atom_37[:-1, 2, :] + carbonyl_to_oxygen * 1.23

    # Now we deal with frames for which there is no next frame available.

    # Calpha to carbonyl both in the current frame. (N, 3)
    calpha_to_carbonyl_term: torch.Tensor = (atom_37[:, 2, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 2, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # Calpha to nitrogen both in the current frame. (N, 3)
    calpha_to_nitrogen_term: torch.Tensor = (atom_37[:, 0, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 0, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    carbonyl_to_oxygen_term: torch.Tensor = (
        calpha_to_carbonyl_term + calpha_to_nitrogen_term
    )  # (N, 3)
    carbonyl_to_oxygen_term = carbonyl_to_oxygen_term / (
        torch.norm(carbonyl_to_oxygen_term, dim=1, keepdim=True) + 1e-7
    )

    # Create a mask that is 1 when the next residue is not available either
    # due to this frame being the C-terminus or the next residue is not
    # known due to pos_is_known being false.

    if pos_is_known is None:
        pos_is_known = torch.ones(
            (atom_37.shape[0],), dtype=torch.int64, device=atom_37.device
        )

    next_res_gone: torch.Tensor = ~pos_is_known.bool()  # (N,)
    next_res_gone = torch.cat(
        [next_res_gone, torch.ones((1,), device=pos_is_known.device).bool()], dim=0
    )  # (N+1, )
    next_res_gone = next_res_gone[1:]  # (N,)

    atom_37[next_res_gone, 4, :] = (
        atom_37[next_res_gone, 2, :] + carbonyl_to_oxygen_term[next_res_gone, :] * 1.23
    )

    return atom_37


def trans_to_atom37(trans):
    B, L, _ = trans.shape
    atom37 = trans.new_zeros(B, L, 37, 3)
    atom37[:, :, 1] = trans
    return atom37


def transrot_to_atom37(rigids):
    atom37_traj = []
    B, L = rigids.shape

    atom37 = compute_backbone(rigids, torch.zeros((B, L, 2), device=rigids.device))[0]
    for i in range(B):
        atom37[i] = adjust_oxygen_pos(atom37[i], None)
    return atom37


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


def get_aligned_rmsd(pos_1, pos_2):
    aligned_pos_1 = rigid_transform_3D(pos_1, pos_2)[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))


def create_full_prot(
    atom37: np.ndarray,
    aatype=None,
    b_factors=None,
):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    atom37_mask = (np.sum(np.abs(atom37), axis=-1) > 1e-7) & (
        np.sum(np.abs(atom37[:, 1:2]), axis=-1) > 1e-7
    )
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        b_factors=b_factors,
    )


def frames_to_pdb(frames, path):
    prots = []
    atom37 = transrot_to_atom37(frames).cpu().numpy()
    for i, pos in enumerate(atom37):
        prots.append(create_full_prot(pos))
    with open(path, "w") as f:
        f.write(prots_to_pdb(prots))


def trans_to_pdb(trans, path):
    prots = []
    atom37 = trans_to_atom37(trans).cpu().numpy()
    for i, pos in enumerate(atom37):
        prots.append(create_full_prot(pos))
    with open(path, "w") as f:
        f.write(prots_to_pdb(prots))


def atom37_to_pdb(atom37, path, mask=None):
    prots = []
    for i, pos in enumerate(atom37):
        if mask is not None:
            pos = pos[mask[i]]
        prots.append(create_full_prot(pos))
    with open(path, "w") as f:
        f.write(prots_to_pdb(prots))


def prots_to_pdb(prots):
    ss = ""
    for i, prot in enumerate(prots):
        ss += f"MODEL {i}\n"
        prot = protein.to_pdb(prot)
        ss += "\n".join(prot.split("\n")[1:-2])
        ss += "\nENDMDL\n"
    return ss


def compute_lddt(pos1, pos2, mask, cutoff=15.0, eps=1e-10, symmetric=False):
    dmat1 = torch.sqrt(
        eps + torch.sum((pos1[..., None, :] - pos1[..., None, :, :]) ** 2, axis=-1)
    )
    dmat2 = torch.sqrt(
        eps + torch.sum((pos2[..., None, :] - pos2[..., None, :, :]) ** 2, axis=-1)
    )
    if symmetric:
        dists_to_score = (dmat1 < cutoff) | (dmat2 < cutoff)
    else:
        dists_to_score = dmat1 < cutoff
    dists_to_score = (
        dists_to_score
        * mask.unsqueeze(-2)
        * mask.unsqueeze(-1)
        * (1.0 - torch.eye(mask.shape[-1]).to(mask))
    )
    dist_l1 = torch.abs(dmat1 - dmat2)
    score = (
        (dist_l1[..., None] < torch.tensor([0.5, 1.0, 2.0, 4.0]).to(pos1))
        .float()
        .mean(-1)
    )
    score = (dists_to_score * score).sum((-1, -2)) / dists_to_score.sum((-1, -2))

    return score


def compute_fape(
    pred_frames,
    target_frames,
    frames_mask,
    pred_positions,
    target_positions,
    positions_mask,
    length_scale,
    l1_clamp_distance=None,
    thresh=None,
    eps=1e-8,
) -> torch.Tensor:
    """
    Computes FAPE loss.

    Args:
        pred_frames:
            [*, N_frames] Rigid object of predicted frames
        target_frames:
            [*, N_frames] Rigid object of ground truth frames
        frames_mask:
            [*, N_frames] binary mask for the frames
        pred_positions:
            [*, N_pts, 3] predicted atom positions
        target_positions:
            [*, N_pts, 3] ground truth positions
        positions_mask:
            [*, N_pts] positions mask
        length_scale:
            Length scale by which the loss is divided
        pair_mask:
            [*,  N_frames, N_pts] mask to use for
            separating intra- from inter-chain losses.
        l1_clamp_distance:
            Cutoff above which distance errors are disregarded
        eps:
            Small value used to regularize denominators
    Returns:
        [*] loss tensor
    """

    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale

    if thresh is not None:
        thresh_mask = torch.sqrt(torch.sum(local_target_pos**2, dim=-1)) < thresh
        mask = thresh_mask * frames_mask[..., None] * positions_mask[..., None, :]

        normed_error = normed_error * mask
        normed_error = torch.sum(normed_error, dim=(-1, -2))
        normed_error = normed_error / (eps + torch.sum(mask, dim=(-1, -2)))

    else:
        normed_error = normed_error * frames_mask[..., None]
        normed_error = normed_error * positions_mask[..., None, :]

        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error
