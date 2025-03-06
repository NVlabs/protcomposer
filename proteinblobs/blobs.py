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
import networkx as nx
import numpy as np
from scipy.special import logsumexp, softmax
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import wishart

alpha_slope = 0.22213567936924072
alpha_intercept = 7.512307373103329

beta_slope = 0.2727479880396384
beta_intercept = 8.28561006547174


def blobify(pos, dssp, radius_thresh=5, size_thresh=5):
    distmat = np.square(pos[:, None] - pos[None]).sum(-1) ** 0.5
    G = nx.Graph()
    edges = np.argwhere((distmat < radius_thresh) & (dssp[None] == dssp[:, None]))
    G.add_edges_from(edges)
    blobs = []
    for con in nx.connected_components(G):
        con = list(con)
        if dssp[con[0]] == 0:
            continue
        if len(con) < size_thresh:
            continue
        blobs.append(
            {
                "residues": con,
                "count": len(con),
                "dssp": dssp[con[0]],
                "pos": pos[con].mean(0).astype(float),
                "covar": np.cov(pos[con].T).astype(float),
            }
        )

    return blobs


def shannon_complexity(pos, dssp, thresh=5):
    blobs = blobify(pos, dssp, radius_thresh=thresh)
    counts = np.array([b["count"] for b in blobs])
    probs = counts / counts.sum()
    return -(probs * np.log(probs)).sum()


def score_blobs(centers, covars):
    energy = 0
    for pos, cov in zip(centers, covars):
        relpos = centers - pos
        d = np.einsum("li,ij,lj->l", relpos, np.linalg.inv(cov), relpos) ** 0.5
        d = d[d != 0]
        energy += (1 / d**2).sum()
    return energy


def sample_blobs(k, nu, psi, sigma):
    while True:
        covar = wishart.rvs(nu, psi, size=k)
        if k == 1:
            covar = covar[None]
        pos = np.random.randn(k, 3) * sigma
        score = score_blobs(pos, covar)
        if np.random.rand() < np.exp(-score):
            return pos, covar


def gmm_ll(pos, centers, covars, probs):
    probs = probs / probs.sum()
    relpos = pos[:, None] - centers[None]
    ll = (
        -0.5 * np.einsum("lni,nij,lnj->ln", relpos, np.linalg.inv(covars), relpos)
        - 0.5 * np.linalg.slogdet(covars)[1][None]
        + np.log(probs)[None]
    )
    return logsumexp(ll, -1)


def dssp_gmm_ll(pos, centers, covars, probs, pos_dssp, centers_dssp):
    if np.any(centers_dssp == 1):
        gmm_ll_1 = gmm_ll(
            pos,
            centers[centers_dssp == 1],
            covars[centers_dssp == 1],
            probs[centers_dssp == 1],
        )
        gmm_ll_1[pos_dssp != 1] = -np.inf
    else:
        gmm_ll_1 = -np.inf * np.ones(len(pos))

    if np.any(centers_dssp == 2):
        gmm_ll_2 = gmm_ll(
            pos,
            centers[centers_dssp == 2],
            covars[centers_dssp == 2],
            probs[centers_dssp == 2],
        )
        gmm_ll_2[pos_dssp != 2] = -np.inf
    else:
        gmm_ll_2 = -np.inf * np.ones(len(pos))

    dssp_1_count = probs[centers_dssp == 1].sum() / probs.sum()
    dssp_2_count = probs[centers_dssp == 2].sum() / probs.sum()

    return np.log(np.exp(gmm_ll_1) * dssp_1_count + np.exp(gmm_ll_2) * dssp_2_count)


"""
Treats the blobs as a GMM and evaluates the average log-DENSITY at each pos.
Note that DENSITY is N * PDF where N is the total number of residues in the blobs.
Not suitable for partial blobs.
"""


def blob_likelihood(pos, dssp, blobs, structured_only=False, mean=True):
    if structured_only:
        pos = pos[dssp != 0]
        dssp = dssp[dssp != 0]
        if len(dssp) == 0:
            return -10
    if len(blobs) == 0:
        ll = -np.inf * np.ones(pos.shape[0])
    else:
        covars = np.stack([blob["covar"] for blob in blobs])
        counts = np.array([blob["count"] for blob in blobs])
        centers = np.stack([blob["pos"] for blob in blobs])

        ll = gmm_ll(pos, centers, covars, counts)
        ll += np.log(counts.sum())

    if not mean:
        return ll
    return ll.mean()


"""
Reblobs the protein and evaluates the number of residues per blob
"""


def blobs_per_res(pos, dssp, thresh=5):
    blobs = blobify(pos, dssp, thresh)
    return len(blobs) / len(pos)


def res_per_blob(pos, dssp, thresh=5):
    blobs = blobify(pos, dssp, thresh)
    return len(pos) / len(blobs)


"""
Reblobs the protein and evaluates the JSD between the GMMs 
defined by the original and new blobs.
Not suitable for partial blobs.
If use_dssp=True, consider sample space R^3 x {1,2}
"""


def reblob_jsd(pos, dssp, blobs, num_samples=1000, use_dssp=False, thresh=5):
    p_blobs = blobs
    q_blobs = blobify(pos, dssp, thresh)

    p_covars = np.stack([blob["covar"] for blob in p_blobs])
    p_counts = np.array([blob["count"] for blob in p_blobs])
    p_centers = np.stack([blob["pos"] for blob in p_blobs])
    p_dssp = np.array([blob["dssp"] for blob in p_blobs])

    q_covars = np.stack([blob["covar"] for blob in q_blobs])
    q_counts = np.array([blob["count"] for blob in q_blobs])
    q_centers = np.stack([blob["pos"] for blob in q_blobs])
    q_dssp = np.array([blob["dssp"] for blob in q_blobs])

    p_idx = np.random.choice(
        np.arange(len(p_blobs)), size=num_samples, p=p_counts / p_counts.sum()
    )
    p_samps = MultivariateNormal(
        torch.from_numpy(p_centers[p_idx]).float(),
        torch.from_numpy(p_covars[p_idx]).float(),
    ).sample()
    if use_dssp:
        p_p = dssp_gmm_ll(p_samps, p_centers, p_covars, p_counts, p_dssp[p_idx], p_dssp)
        p_q = dssp_gmm_ll(p_samps, q_centers, q_covars, q_counts, p_dssp[p_idx], q_dssp)

    else:
        p_p = gmm_ll(p_samps, p_centers, p_covars, p_counts)
        p_q = gmm_ll(p_samps, q_centers, q_covars, q_counts)

    p_m = np.log(np.exp(p_p) / 2 + np.exp(p_q) / 2)
    kl_p = (p_p - p_m).mean()

    q_idx = np.random.choice(
        np.arange(len(q_blobs)), size=num_samples, p=q_counts / q_counts.sum()
    )
    q_samps = MultivariateNormal(
        torch.from_numpy(q_centers[q_idx]).float(),
        torch.from_numpy(q_covars[q_idx]).float(),
    ).sample()

    if use_dssp:
        q_p = dssp_gmm_ll(q_samps, p_centers, p_covars, p_counts, q_dssp[q_idx], p_dssp)
        q_q = dssp_gmm_ll(q_samps, q_centers, q_covars, q_counts, q_dssp[q_idx], q_dssp)
    else:
        q_p = gmm_ll(q_samps, p_centers, p_covars, p_counts)
        q_q = gmm_ll(q_samps, q_centers, q_covars, q_counts)
    q_m = np.log(np.exp(q_p) / 2 + np.exp(q_q) / 2)
    kl_q = (q_q - q_m).mean()

    return (kl_p + kl_q) / 2


"""
Calculate the sum of the difference in the fraction of residues
assigned to a blob and the fraction of total residues actually placed in the blob.
"""


def blob_misplacement(pos, dssp, blobs, thresh=2.25, structured_only=False):
    if structured_only:
        pos = pos[dssp != 0]
        dssp = dssp[dssp != 0]
        if len(dssp) == 0:
            return 1

    assigned = []
    actual = []
    for b in blobs:
        relpos = pos - b["pos"]
        dists = np.einsum("li,ij,lj->l", relpos, np.linalg.inv(b["covar"]), relpos)
        mask = dists**0.5 < thresh
        actual.append(mask.sum())
        assigned.append(b["count"])

    assigned = np.array(assigned) / np.sum(assigned)
    actual = np.array(actual) / np.sum(actual)
    return np.sum(np.abs(assigned - actual))


"""
The fraction of residues within some Mahalanobis distance of each blob
with the correct secondary structure. 
Suitable for partial blobs.
"""


def blob_accuracy(pos, dssp, blobs, thresh=2.25, structured_only=False):
    if structured_only:
        pos = pos[dssp != 0]
        dssp = dssp[dssp != 0]
        if len(dssp) == 0:
            return 0

    correct, total = 0, 0

    for b in blobs:
        relpos = pos - b["pos"]
        dists = np.einsum("li,ij,lj->l", relpos, np.linalg.inv(b["covar"]), relpos)
        mask = dists**0.5 < thresh
        correct += (dssp[mask] == b["dssp"]).sum()
        total += mask.sum()

    return correct / total


"""
Based on sheet and helix GMMs, computes p(ss | pos) for each pos in the new protein.
Reports the average p(ss | pos) over all non-loop residues.
Not suitable for partial blobs.
"""


def soft_blob_accuracy(pos, dssp, blobs, structured_only=False):
    if structured_only:
        pos = pos[dssp != 0]
        dssp = dssp[dssp != 0]
        if len(dssp) == 0:
            return 0

    sheet_ll = blob_likelihood(
        pos, dssp, [b for b in blobs if b["dssp"] == 1], mean=False
    )
    helix_ll = blob_likelihood(
        pos, dssp, [b for b in blobs if b["dssp"] == 2], mean=False
    )

    probs = softmax(np.stack([sheet_ll, helix_ll], -1), -1)
    return ((probs[:, 0][dssp == 1]).sum() + (probs[:, 1][dssp == 2]).sum()) / len(dssp)


"""
Reports the fraction of residues within some Mahalanobis distance of some blob.
Not suitable for partial blobs.
"""


def blob_coverage(pos, dssp, blobs, thresh=2.25, structured_only=False):
    if structured_only:
        pos = pos[dssp != 0]
        dssp = dssp[dssp != 0]
        if len(dssp) == 0:
            return 0

    idx = np.arange(len(dssp))
    seen = []

    for b in blobs:
        relpos = pos - b["pos"]
        dists = np.einsum("li,ij,lj->l", relpos, np.linalg.inv(b["covar"]), relpos)
        mask = dists**0.5 < thresh
        seen.extend(idx[mask])

    seen = list(set(seen))
    total = len(dssp)
    return len(seen) / total


"""
Counts the number of residues within some Mahalanobis distance of some blob,
relative to the total number expected for all blobs collectively.
If unique=True, then residues within cutoff of multiple blobs will only be counted once.
Suitable for partial blobs.
"""


def blob_occupancy(pos, dssp, blobs, thresh=2.25, structured_only=False, unique=False):
    if structured_only:
        pos = pos[dssp != 0]
        dssp = dssp[dssp != 0]
        if len(dssp) == 0:
            return 0

    idx = np.arange(len(dssp))
    seen = []

    for b in blobs:
        relpos = pos - b["pos"]
        dists = np.einsum("li,ij,lj->l", relpos, np.linalg.inv(b["covar"]), relpos)
        mask = dists**0.5 < thresh
        seen.extend(idx[mask])

    if unique:
        seen = list(set(seen))
    total = sum(b["count"] for b in blobs)
    return len(seen) / total
