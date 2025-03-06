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
import pickle
import pydssp
import math
import numpy as np
import torch
import pickle
from openfold.np import protein
from torch.utils.data import default_collate
from sklearn.cluster import KMeans
import pandas as pd
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.np.residue_constants import restype_order, atom_order
from scipy.spatial.transform import Rotation as spRotation
from .blobs import blobify
import proteinblobs.blobs as blobs
import proteinblobs.multiflow.datasets as mfdatasets
from pathlib import Path


def from_3_points(
    p_neg_x_axis: torch.Tensor,
    origin: torch.Tensor,
    p_xy_plane: torch.Tensor,
    eps: float = 1e-8,
) -> Rigid:
    p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
    origin = torch.unbind(origin, dim=-1)
    p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

    e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
    e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

    the_sum = sum((c * c for c in e0))
    the_other_sum = the_sum + eps

    ###### Workaround to prevent unexplicable error when taking sqrt with torch.
    sum_np = the_other_sum.numpy()
    sqrt = np.sqrt(sum_np)
    # denom = torch.sqrt(the_other_sum)
    denom = torch.from_numpy(sqrt)

    e0 = [c / denom for c in e0]
    dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
    e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
    denom = torch.sqrt(sum((c * c for c in e1)) + eps)
    e1 = [c / denom for c in e1]
    e2 = [
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
    ]
    rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
    rots = rots.reshape(rots.shape[:-1] + (3, 3))
    rot_obj = Rotation(rot_mats=rots, quats=None)

    return Rigid(rot_obj, torch.stack(origin, dim=-1))


def prot_to_frames(ca_coords, c_coords, n_coords):
    prot_frames = from_3_points(
        torch.from_numpy(c_coords),
        torch.from_numpy(ca_coords),
        torch.from_numpy(n_coords),
    )
    rots = torch.eye(3)
    rots[0, 0] = -1
    rots[2, 2] = -1
    rots = Rotation(rot_mats=rots)
    frames = prot_frames.compose(Rigid(rots, None))
    return frames


class SeqCollate:
    def __init__(self, args):
        self.args = args
        if self.args.no_crop or self.args.no_pad:
            self.seq_len_keys = [
                "grounding_feat",
                "grounding_pos",
                "grounding_mask",
                "chain_idx",
                "res_idx",
                "seqres",
                "res_mask",
                "mask",
                "rots",
                "trans",
            ]
        else:
            self.seq_len_keys = [
                "grounding_feat",
                "grounding_pos",
                "grounding_mask",
            ]

    def __call__(self, batch):
        seq_len_batch = {}
        for key in self.seq_len_keys:
            elems = [item[key] for item in batch]
            max_L = max([len(elem) for elem in elems])
            mask = torch.zeros((len(elems), max_L), dtype=torch.int16)
            elem_tensor = []
            for i, elem in enumerate(elems):
                L = len(elem)
                if isinstance(elem, torch.Tensor):
                    elem = elem.numpy()
                elem = np.concatenate(
                    [elem, np.zeros((max_L - L, *elem.shape[1:]), dtype=elem.dtype)],
                    axis=0,
                )
                elem_tensor.append(elem)
                mask[i, :L] = 1
            seq_len_batch[key] = torch.from_numpy(np.stack(elem_tensor, axis=0))
            seq_len_batch[f"{key}_mask"] = mask

        # remove all self.seq_len_keys from batch and put it through default collate
        for item in batch:
            for key in self.seq_len_keys:
                del item[key]

        batch = default_collate(batch)
        batch.update(seq_len_batch)
        return batch


class BlobDataset(torch.utils.data.Dataset):
    def __init__(self, blobs):
        self.blobs = blobs

    def __len__(self):
        return len(self.blobs)


class StructureDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.length_dist_npz is not None:
            lens = np.load(self.args.length_dist_npz)["lengths"]
            sample_lens = []
            for i in range(100000):
                choice = 0
                while choice < 4 or choice > args.crop:
                    choice = np.random.choice(lens)
                sample_lens.append(choice)
            self.length_dist = np.array(sample_lens)
        if self.args.use_latents:
            latents_names = pickle.load(open(self.args.latents_order, "rb"))
            latents_names = [Path(f).stem for f in latents_names]
            latents_tensors = torch.load(self.args.latents_path)
            self.latents = {
                latents_names[i].lower(): latents_tensors[i]
                for i in range(len(latents_names))
            }

    def __len__(self):
        return None

    def __getitem__(self, idx):
        return None

    def process_prot(self, idx, name, pdb=None, prot=None):
        if prot is None:
            prot = protein.from_pdb_string(pdb)
        atom37 = prot.atom_positions.astype(np.float32)
        frames = prot_to_frames(
            ca_coords=prot.atom_positions[:, atom_order["CA"]],
            c_coords=prot.atom_positions[:, atom_order["C"]],
            n_coords=prot.atom_positions[:, atom_order["N"]],
        )
        res_mask = np.ones(atom37.shape[0], dtype=np.float32)
        seqres = prot.aatype.astype(int)
        res_idx = prot.residue_index
        chain_idx = np.zeros_like(seqres)
        return self.item_from_prot(
            name, atom37, frames, res_mask, seqres, res_idx, chain_idx
        )

    def item_from_prot(
        self, name, atom37, frames, res_mask, seqres, res_idx, chain_idx
    ):
        mask = np.ones(atom37.shape[0], dtype=np.float32)
        # take N, CA, C, and O. The order in atom37 is N, CA, C, CB, O ... (see atom_types in residue_constants.py)
        bb_pos = np.concatenate(
            [atom37[:, :3, :], atom37[:, 4:5, :]], axis=1
        )  # (L, 4, 3)

        L = frames.shape[0]
        ## filter lenght and run pydssp secondary structure determination
        if L < 8:
            print(
                f"Only {L} residues in the protein {name}. Resampling another protein."
            )
            return self.__getitem__(np.random.randint(len(self)))
        try:
            dssp = pydssp.assign(
                bb_pos, out_type="index"
            )  # 0: loop,  1: alpha-helix,  2: beta-strand
        except Exception as e:
            print(
                f"Running pydssp failed in the protein {name}. Resampling another protein."
            )
            print(str(e))
            return self.__getitem__(np.random.randint(len(self)))

        ## crop
        if (not self.args.no_crop) and L > self.args.crop:
            start = np.random.randint(0, L - self.args.crop + 1)
            if self.args.overfit:
                start = 0
            frames = frames[start : start + self.args.crop]
            mask = mask[start : start + self.args.crop]
            res_mask = res_mask[start : start + self.args.crop]
            seqres = seqres[start : start + self.args.crop]
            chain_idx = chain_idx[start : start + self.args.crop]
            res_idx = res_idx[start : start + self.args.crop]
            dssp = dssp[start : start + self.args.crop]

        ## center
        com = (frames._trans * mask[:, None]).sum(0) / mask.sum()
        frames._trans -= com

        ## rotate
        randrot = spRotation.random().as_matrix().astype(np.float32)
        randrot = torch.from_numpy(randrot)
        frames._trans = frames._trans @ randrot.T
        frames._rots._rot_mats = randrot @ frames._rots._rot_mats

        ## label
        thresh = (
            np.random.rand() * (self.args.max_blob_thresh - self.args.min_blob_thresh)
            + self.args.min_blob_thresh
        )
        blobs = blobify(frames._trans.numpy(), dssp, thresh)

        if not blobs:
            print(f"No blobs in the protein {name}. Resampling another protein.")
            return self.__getitem__(np.random.randint(len(self)))

        if self.args.blob_drop_prob > 0.0 and len(blobs) > 1:
            np.random.shuffle(blobs)
            n = np.random.randint(1, len(blobs))
            blobs = blobs[:n]

        ## pad
        if (not self.args.no_pad) and L < self.args.crop:
            pad = self.args.crop - L
            frames = Rigid.cat(
                [frames, Rigid.identity((pad,), requires_grad=False, fmt="rot_mat")], 0
            )

            mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
            res_mask = np.concatenate([res_mask, np.zeros(pad, dtype=np.float32)])
            seqres = np.concatenate([seqres, np.zeros(pad, dtype=int)])
            res_idx = np.concatenate([res_idx, np.zeros(pad, dtype=int)])
            chain_idx = np.concatenate([chain_idx, np.zeros(pad, dtype=int)])

        ## featurize
        if self.args.synthetic_blobs:
            grounding_pos, grounding_feat, grounding_mask = self.get_synthetic_blobs()
            prot_size = int(grounding_feat[:, 1].sum())
            mask[prot_size:] = 0
            res_mask[prot_size:] = 0
            mask[:prot_size] = 1
            res_mask[:prot_size] = 1
        else:
            grounding_pos = []
            grounding_feat = []
            grounding_covar = []
            for blob in blobs:
                grounding_pos.append(blob["pos"])
                grounding_feat.append((blob["dssp"], blob["count"]))
                grounding_covar.append(blob["covar"].flatten())
            grounding_pos = np.array(grounding_pos).astype(np.float32)
            grounding_feat = np.array(grounding_feat)
            grounding_covar = np.array(grounding_covar).astype(np.float32)
            grounding_feat = np.concatenate(
                [grounding_feat, grounding_covar], axis=-1
            ).astype(np.float32)
            grounding_mask = np.ones_like(grounding_feat[:, 0])

        if self.args.fixed_inference_size is not None:
            assert self.args.fixed_inference_size <= self.args.crop
            mask[self.args.fixed_inference_size :] = 0
            res_mask[self.args.fixed_inference_size :] = 0
            mask[: self.args.fixed_inference_size] = 1
            res_mask[: self.args.fixed_inference_size] = 1

        if self.args.length_dist_npz is not None:
            prot_size = np.random.choice(self.length_dist)
            mask[prot_size:] = 0
            res_mask[prot_size:] = 0
            mask[:prot_size] = 1
            res_mask[:prot_size] = 1

        if self.args.use_latents:
            try:
                latents = self.latents[name.lower()]
            except Exception as e:
                print(f"key error for  {name.lower()}")
                return self.__getitem__(np.random.randint(len(self)))
        else:
            latents = 0

        return {
            "name": name,
            "grounding_pos": grounding_pos,
            "grounding_feat": grounding_feat,
            "grounding_mask": grounding_mask,
            "trans": frames._trans,
            "rots": frames._rots._rot_mats,
            "mask": mask,
            "res_mask": res_mask,
            "seqres": seqres,
            "res_idx": res_idx,
            "chain_idx": chain_idx,
            "blobs": pickle.dumps(blobs),
            "thresh": thresh,
            "latents": latents,
        }

    def get_synthetic_blobs(self):
        pos, covar = blobs.sample_blobs(
            self.args.num_blobs,
            nu=self.args.nu,
            psi=(1 / self.args.nu) * self.args.psi**2 * np.eye(3),
            sigma=self.args.sigma,
        )
        is_helix = np.random.rand(self.args.num_blobs) < self.args.helix_frac
        volume = np.linalg.det(covar) ** 0.5

        counts = np.where(
            is_helix,
            blobs.alpha_slope * volume + blobs.alpha_intercept,
            blobs.beta_slope * volume + blobs.beta_intercept,
        ).astype(int)

        dssp = np.where(is_helix, 1, 2)

        grounding_pos = pos[None].astype(np.float32)
        grounding_feat = np.zeros((1, self.args.num_blobs, 11), dtype=np.float32)
        grounding_feat[:, :, 0] = dssp  # all helix for now
        grounding_feat[:, :, 1] = counts
        grounding_feat[:, :, 2:] = covar.reshape(1, self.args.num_blobs, 9)
        grounding_mask = np.ones(self.args.num_blobs, dtype=np.float32)
        return grounding_pos[0], grounding_feat[0], grounding_mask


class MultiflowDataset(StructureDataset):
    def __init__(self, args, dataset_cfg):
        super().__init__(args)
        self.args = args
        self.mf_dataset = mfdatasets.PdbDataset(
            dataset_cfg=dataset_cfg, task="hallucination", is_training=True
        )

    def __len__(self):
        if self.args.dataset_multiplicity is not None:
            return len(self.mf_dataset.csv) * self.args.dataset_multiplicity
        else:
            return len(self.mf_dataset.csv)

    def __getitem__(self, idx):
        if self.args.overfit:
            idx = 0
        if self.args.dataset_multiplicity is not None:
            idx = idx % self.args.dataset_multiplicity
        row = self.mf_dataset.csv.iloc[idx]
        name = row["pdb_name"]
        path = f"{self.args.pkl_dir}/{row['processed_path'].replace('./', '')}"

        # res_idx, chain_idx, res_mask, trans_1, rotmats_1, aatypes_1, res_plddt
        mf_prot = mfdatasets._process_csv_row(path)

        atom37 = mf_prot["all_atom_positions"].float().numpy()
        # atom37_to_pdb(atom37[None], f'../protfid/data/mf_testset/{name}.pdb')
        frames = Rigid(
            Rotation(rot_mats=mf_prot["rotmats_1"].float(), quats=None),
            mf_prot["trans_1"].float(),
        )

        # take N, CA, C, and O. The order in atom37 is N, CA, C, CB, O ... (see atom_types in residue_constants.py)
        bb_pos = np.concatenate(
            [atom37[:, :3, :], atom37[:, 4:5, :]], axis=1
        )  # (L, 4, 3)

        L = frames.shape[0]
        res_mask = mf_prot["res_mask"].float().numpy()
        seqres = mf_prot["aatypes_1"].numpy()
        res_idx = mf_prot["res_idx"]
        chain_idx = mf_prot["chain_idx"]
        return self.item_from_prot(
            name, atom37, frames, res_mask, seqres, res_idx, chain_idx
        )


class GenieDBDataset(StructureDataset):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        files = os.listdir(self.args.genie_db_path)
        self.db = [file for file in files if ".pdb" in file]
        self.db = self.db * args.repeat_dataset

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        if self.args.overfit:
            idx = 0
        name = self.db[idx]
        with open(f"{self.args.genie_db_path}/{name}", "r") as f:
            pdb = f.read()
        try:
            sample = self.process_prot(idx, name, pdb)
        except Exception as e:
            # for some reason the name printing here does not work so dont rely on it. We instead write it to a file.
            print("name", name, flush=True)
            with open(os.path.join(os.environ["MODEL_DIR"], "debug.txt"), "w") as f:
                f.write(name)
            raise e
        return sample


class SCOPDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        names = os.listdir(args.scop_dir)
        nmr_names = pd.read_csv(args.scop_nmr_csv)
        self.names = [
            name for name in names if name not in nmr_names["nmr_names"].tolist()
        ]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.args.overfit:
            idx = 0

        name = self.names[idx]
        pdb = open(f"{self.args.scop_dir}/{name}", "r").read()
        prot = protein.from_pdb_string(pdb)

        atom37 = prot.atom_positions.astype(np.float32)
        # take N, CA, C, and O. The order in atom37 is N, CA, C, CB, O ... (see atom_types in residue_constants.py)
        bb_pos = np.concatenate(
            [atom37[:, :3, :], atom37[:, 4:5, :]], axis=1
        )  # (L, 4, 3)
        dssp = pydssp.assign(
            bb_pos, out_type="index"
        )  # 0: loop,  1: alpha-helix,  2: beta-strand

        pos = atom37[:, 1, :]
        L = atom37.shape[0]
        mask = np.ones(L, dtype=np.float32)
        seqres = prot.aatype.astype(int)
        resid = prot.residue_index.astype(int)

        cm = (pos * mask[:, None]).sum(0, keepdims=True) / mask.sum()
        pos = pos - cm
        if not self.args.overfit_rot:
            rot = spRotation.random().as_matrix().astype(np.float32)
            pos = pos @ rot.T

        n_clusters = math.ceil(L / self.args.res_per_cluster)
        kmeans = KMeans(n_clusters=n_clusters, n_init=1).fit(bb_pos[:, 1])
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        dssp_onehot = torch.nn.functional.one_hot(torch.from_numpy(dssp), num_classes=3)

        dssp_count = torch.zeros(len(centers), 3, dtype=torch.long).scatter_add_(
            dim=0,
            index=torch.from_numpy(labels).long()[:, None].expand(-1, 3),
            src=dssp_onehot,
        )
        dssp_dist = dssp_count / dssp_count.sum(1)[:, None]

        return {
            "prot": prot,
            "labels": labels,
            "centers": centers,
            "dssp_dist": dssp_dist,
            "resid": resid,
            "name": name,
            "dssp": dssp,
            "bb_pos": pos,
            "mask": mask,
            "seqres": seqres,
        }
