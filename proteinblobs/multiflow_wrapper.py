# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from .logger import get_logger

logger = get_logger(__name__)

from proteinblobs.visualize import visualize_blobs

import torch, time, os
from omegaconf import OmegaConf
import numpy as np
import pickle
from openfold.utils.rigid_utils import Rigid, Rotation
from .multiflow.flow_model import FlowModel
from .multiflow.data.interpolant import Interpolant
from .multiflow.data import utils as du
from .multiflow.data import so3_utils, all_atom
from .utils import frames_to_pdb
from proteinblobs.utils import atom37_to_pdb
from proteinblobs.designability_utils import run_designability
from .blobs import (
    blob_likelihood,
    reblob_jsd,
    blob_accuracy,
    soft_blob_accuracy,
    blob_coverage,
    blob_occupancy,
    blob_misplacement,
)
from .wrapper import Wrapper
import copy
import pydssp


class MultiflowWrapper(Wrapper):
    def __init__(self, args, cfg=None):
        super().__init__(args)

        cfg = cfg or OmegaConf.load(args.multiflow_yaml)
        if "guidance" not in args.__dict__:
            args.guidance = False
        if "use_latents" not in args.__dict__:
            args.use_latents = False
        if "interpolate_sc" not in args.__dict__:
            args.interpolate_sc = False
        if "guidance_weight" not in args.__dict__:
            args.guidance_weight = 1.0
        if "inference_rot_scaling" not in args.__dict__:
            args.inference_rot_scaling = 10
        if "inference_gating" not in args.__dict__:
            args.inference_gating = 1.0

        cfg.interpolant.rots.exp_rate = args.inference_rot_scaling
        cfg.inference.interpolant.rots.exp_rate = args.inference_rot_scaling
        cfg.inference.interpolant.sampling.num_timesteps = args.num_timesteps
        cfg.model.edge_features.self_condition = (
            args.self_condition
        )  # only the cfg.interpolant.self_condition is used anywhere in the code
        cfg.interpolant.self_condition = args.self_condition
        self.cfg = cfg

        self.model = FlowModel(cfg.model, args)
        if self.args.finetune:
            ckpt = torch.load(self.args.pretrained_mf_path, map_location=self.device)
            self.load_state_dict(ckpt["state_dict"], strict=False)
        self.interpolant = Interpolant(cfg.interpolant, args)
        self.model_uncond = None
        self.aatype_pred_num_tokens = cfg.model.aatype_pred_num_tokens
        self._exp_cfg = cfg.experiment
        self.prots = []
        self.masks = []

    def general_step(self, batch):
        self.iter_step += 1
        self.interpolant.set_device(self.device)
        if self.args.only_save_blobs:
            return None

        ## translation
        batch["trans_1"] = batch["trans"]
        batch["rotmats_1"] = batch["rots"]
        batch["aatypes_1"] = batch["seqres"]
        if "diffuse_mask" not in batch:
            batch["diffuse_mask"] = torch.ones_like(batch["res_mask"])
        noisy_batch = self.interpolant.corrupt_batch(batch)
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch["mask"] * noisy_batch["diffuse_mask"]

        # if training_cfg.mask_plddt:
        #     loss_mask *= noisy_batch['plddt_mask']

        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError("Empty batch encountered")
        num_batch, num_res = loss_mask.shape
        # Ground truth labels
        gt_trans_1 = noisy_batch["trans_1"]
        gt_rotmats_1 = noisy_batch["rotmats_1"]
        gt_aatypes_1 = noisy_batch["aatypes_1"]
        rotmats_t = noisy_batch["rotmats_t"]
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3]

        # Timestep used for normalization.
        r3_t = noisy_batch["r3_t"]  # (B, 1)
        so3_t = noisy_batch["so3_t"]  # (B, 1)
        cat_t = noisy_batch["cat_t"]  # (B, 1)
        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)
        )  # (B, 1, 1)
        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)
        )  # (B, 1, 1)
        if training_cfg.aatypes_loss_use_likelihood_weighting:
            cat_norm_scale = 1 - torch.min(
                cat_t, torch.tensor(training_cfg.t_normalize_clip)
            )  # (B, 1)
            assert cat_norm_scale.shape == (num_batch, 1)
        else:
            cat_norm_scale = 1.0
        # Model output predictions.
        if self.args.self_condition and np.random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch["trans_sc"] = model_sc["pred_trans"] * noisy_batch[
                    "diffuse_mask"
                ][..., None] + noisy_batch["trans_1"] * (
                    1 - noisy_batch["diffuse_mask"][..., None]
                )
                logits_1 = torch.nn.functional.one_hot(
                    batch["aatypes_1"].long(), num_classes=self.aatype_pred_num_tokens
                ).float()
                noisy_batch["aatypes_sc"] = model_sc["pred_logits"] * noisy_batch[
                    "diffuse_mask"
                ][..., None] + logits_1 * (1 - noisy_batch["diffuse_mask"][..., None])
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output["pred_trans"]
        pred_rotmats_1 = model_output["pred_rotmats"]
        pred_logits = model_output["pred_logits"]  # (B, N, aatype_pred_num_tokens)
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError("NaN encountered in pred_rots_vf")
        # aatypes loss
        ce_loss = (
            torch.nn.functional.cross_entropy(
                pred_logits.reshape(-1, self.aatype_pred_num_tokens),
                gt_aatypes_1.flatten().long(),
                reduction="none",
            ).reshape(num_batch, num_res)
            / cat_norm_scale
        )
        aatypes_loss = torch.sum(ce_loss * loss_mask, dim=-1) / (loss_denom / 3)
        aatypes_loss *= training_cfg.aatypes_loss_weight
        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        bb_atom_loss = (
            torch.sum(
                (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
                dim=(-1, -2, -3),
            )
            / loss_denom
        )

        # Translation VF loss
        trans_error = (
            (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
        )
        trans_loss = (
            training_cfg.translation_loss_weight
            * torch.sum(trans_error**2 * loss_mask[..., None], dim=(-1, -2))
            / loss_denom
        )
        trans_loss = torch.clamp(trans_loss, max=5)
        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        rots_vf_loss = (
            training_cfg.rotation_loss_weights
            * torch.sum(rots_vf_error**2 * loss_mask[..., None], dim=(-1, -2))
            / loss_denom
        )

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * 3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * 3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * 3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * 3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
        )
        dist_mat_loss /= torch.sum(pair_dist_mask, dim=(1, 2)) + 1

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (
            bb_atom_loss * training_cfg.aux_loss_use_bb_loss
            + dist_mat_loss * training_cfg.aux_loss_use_pair_loss
        )
        auxiliary_loss *= (r3_t[:, 0] > training_cfg.aux_loss_t_pass) & (
            so3_t[:, 0] > training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        auxiliary_loss = torch.clamp(auxiliary_loss, max=5)

        loss = trans_loss + rots_vf_loss + auxiliary_loss + aatypes_loss
        if torch.any(torch.isnan(loss)):
            raise ValueError("NaN loss encountered")
        self.log("loss", loss)
        self.log("trans_loss", trans_loss)
        self.log("auxiliary_loss", auxiliary_loss)
        self.log("rots_vf_loss", rots_vf_loss)
        self.log("aatypes_loss", aatypes_loss)
        self.log("dur", time.time() - self.last_log_time)
        for name in batch["name"]:
            self.log("name", name)
        self.last_log_time = time.time()
        return loss.mean()

    def validation_step_extra(self, batch, batch_idx):
        grounding_pos = batch["grounding_pos"].cpu()
        grounding_feat = batch["grounding_feat"].cpu()
        grounding_mask = batch["grounding_mask"].cpu()
        grounding_size = grounding_feat[:, :, 1]
        grounding_dssp = grounding_feat[:, :, 0]

        if self.args.inf_batches > batch_idx:
            out_dir = (
                f"{os.environ['MODEL_DIR']}/epoch{self.current_epoch}_batch{batch_idx}"
            )
            os.makedirs(out_dir, exist_ok=True)

            ref_path = f"{out_dir}/ref_gpuidx{self.global_rank}.pdb"
            frames_to_pdb(
                Rigid(trans=batch["trans"], rots=Rotation(rot_mats=batch["rots"])),
                ref_path,
            )

            ### save blobs
            os.makedirs(f"{os.environ['MODEL_DIR']}/last_samples", exist_ok=True)
            for i in range(len(grounding_pos)):
                single_blob_path = f"{os.environ['MODEL_DIR']}/last_samples/{batch['name'][i].split('.')[0]}.npz"
                np.savez(
                    single_blob_path,
                    counts=grounding_size[i][grounding_mask[i].bool()].cpu().numpy(),
                    dssp=grounding_dssp[i][grounding_mask[i].bool()].cpu().numpy(),
                    covar=grounding_feat[i][grounding_mask[i].bool(), -9:]
                    .reshape(-1, 3, 3)
                    .cpu()
                    .numpy(),
                    pos=grounding_pos[i][grounding_mask[i].bool()].cpu().numpy(),
                )
            if self.args.only_save_blobs:
                self.log("loss", np.nan)
                return

            if self.args.ref_as_sample:
                samples = (
                    all_atom.atom37_from_trans_rot(
                        batch["trans"], batch["rots"], batch["res_mask"]
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                samples, _ = self.inference(batch)

            masks = batch["res_mask"].bool().cpu().numpy()
            self.prots.extend([sample[mask] for sample, mask in zip(samples, masks)])

            sample_path = f"{out_dir}/sample_gpuidx{self.global_rank}.pdb"
            atom37_to_pdb(samples, sample_path)

            for i, sample in enumerate(samples):
                single_sample_path = f"{os.environ['MODEL_DIR']}/last_samples/{batch['name'][i].split('.')[0]}.pdb"
                atom37_to_pdb(sample[None], single_sample_path)

            for i in range(len(grounding_pos)):
                pred_prot = open(sample_path).read().split("ENDMDL\nMODEL")[i]
                tmp_path = f"/tmp/tmp_{batch['name'][i]}_gpuidx{self.global_rank}_{self.args.run_name}.pdb"
                open(tmp_path, "w").write(pred_prot)
                ref_prot = open(ref_path).read().split("ENDMDL\nMODEL")[i]
                tmp_ref_path = f"/tmp/tmp_ref_{batch['name'][i]}_gpuidx{self.global_rank}_{self.args.run_name}.pdb"
                open(tmp_ref_path, "w").write(ref_prot)
                visualize_blobs(
                    [tmp_path, tmp_ref_path],
                    grounding_pos[i][grounding_mask[i].bool()].cpu().numpy(),
                    grounding_feat[i][grounding_mask[i].bool(), -9:]
                    .reshape(-1, 3, 3)
                    .cpu()
                    .numpy(),
                    f"{out_dir}/blobs_{batch['name'][i]}_gpuidx{self.global_rank}.pse",
                )

            grounding_path = f"{out_dir}/blobs_gpuidx{self.global_rank}.npz"
            np.savez(
                grounding_path,
                grounding_pos=grounding_pos,
                grounding_feat=grounding_feat,
                grounding_mask=grounding_mask,
            )

            ### Compute grounding accuracy
            bb_pos_gen = np.concatenate(
                [samples[:, :, :3, :], samples[:, :, 4:5, :]], axis=2
            )  # (L, 4, 3)
            dssp_gen = []
            for pos in bb_pos_gen:
                dssp_gen.append(
                    pydssp.assign(pos, out_type="index")
                )  # 0: loop,  1: alpha-helix,  2: beta-strand

            ca_pos_gen = torch.from_numpy(samples[:, :, 1, :])
            dssp_gen = torch.from_numpy(np.stack(dssp_gen))

            ### blob likelihood
            blobss = [pickle.loads(b) for b in batch["blobs"]]

            for i, (pos, mask, dssp, blobs, thresh) in enumerate(
                zip(ca_pos_gen, masks, dssp_gen, blobss, batch["thresh"])
            ):
                pos = pos[mask].numpy()
                dssp = dssp[mask].numpy()

                try:
                    self.log(
                        "reblob_jsd",
                        reblob_jsd(
                            pos, dssp, blobs, thresh=thresh.item(), structured_only=True
                        ),
                    )
                    self.log(
                        "reblob_jsd_dssp",
                        reblob_jsd(
                            pos,
                            dssp,
                            blobs,
                            use_dssp=True,
                            thresh=thresh.item(),
                            structured_only=True,
                        ),
                    )
                    self.log("reblob_success", 1.0)
                except Exception as e:
                    self.log("reblob_jsd", np.nan)
                    self.log("reblob_jsd_dssp", np.nan)
                    self.log("reblob_success", 0.0)
                self.log(
                    "blob_likelihood",
                    blob_likelihood(pos, dssp, blobs, structured_only=True),
                )
                self.log(
                    "misplacement",
                    blob_misplacement(pos, dssp, blobs, structured_only=True),
                )
                self.log(
                    "blob_accuracy",
                    blob_accuracy(pos, dssp, blobs, structured_only=True),
                )
                self.log(
                    "blob_occupancy",
                    blob_occupancy(pos, dssp, blobs, structured_only=True),
                )
                self.log(
                    "soft_blob_accuracy",
                    soft_blob_accuracy(pos, dssp, blobs, structured_only=True),
                )
                self.log(
                    "blob_coverage",
                    blob_coverage(pos, dssp, blobs, structured_only=True),
                )

    def on_validation_epoch_start(self):
        if self.args.guidance:
            torch.cuda.empty_cache()
            args_uncond = copy.deepcopy(self.args)
            args_uncond.freeze_weights = True
            args_uncond.extra_attn_layer = False
            args_uncond.blob_attention = False
            self.model_uncond = FlowModel(self.cfg.model, args_uncond)
            ckpt = torch.load(self.args.pretrained_mf_path, map_location=self.device)
            self.model_uncond.load_state_dict(
                {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
            )
            self.model_uncond = self.model_uncond.to("cuda")

    def on_validation_epoch_end_extra(self):
        if (
            self.args.self_consistency
            and self.trainer.current_epoch % self.args.designability_freq == 0
        ):
            raise Exception("not implemented fully")
            results = run_designability(
                self.prots, None, self.global_rank, sequences=self.seqs
            )

        elif (
            self.args.designability
            and self.trainer.current_epoch % self.args.designability_freq == 0
        ):
            results = run_designability(
                self.prots[: self.args.num_designability_prots],
                self.args.pmpnn_path,
                self.global_rank,
            )

        if (
            self.args.self_consistency
            and self.trainer.current_epoch % self.args.designability_freq == 0
            or self.args.designability
            and self.trainer.current_epoch % self.args.designability_freq == 0
        ):
            self.log("mean_tm_score", results["tm_score"], extend=True)
            self.log("mean_rmsd", results["rmsd"], extend=True)
            self.log("median_tm_score", np.median(results["tm_score"]))
            self.log("median_rmsd", np.median(results["rmsd"]))
            self.log("top_tm_score", results["top_tm_score"], extend=True)
            self.log("top_rmsd", results["top_rmsd"], extend=True)
            self.log("top_median_tm_score", np.median(results["top_tm_score"]))
            self.log("top_median_rmsd", np.median(results["top_rmsd"]))
            self.log("designable", results["top_rmsd"] < 2, extend=True)
            self.log("tm_score>05", results["tm_score"] > 0.5, extend=True)
        self.prots = []
        self.masks = []
        torch.cuda.empty_cache()
        self.model_uncond = None

    def inference(self, batch, trans_0=None, rotmats_0=None):
        interpolant = Interpolant(self.cfg.inference.interpolant, self.args)
        interpolant.set_device(self.device)

        true_bb_pos = None
        trans_1 = rotmats_1 = diffuse_mask = aatypes_1 = true_aatypes = None
        num_batch, sample_length = batch["res_mask"].shape

        prot_traj, model_traj = interpolant.sample(
            batch["res_mask"],
            self.model,
            self.model_uncond,
            grounding_pos=batch["grounding_pos"],
            grounding_feat=batch["grounding_feat"],
            grounding_mask=batch["grounding_mask"],
            trans_0=trans_0,
            rotmats_0=rotmats_0,
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            aatypes_1=aatypes_1,
            diffuse_mask=diffuse_mask,
            forward_folding=False,
            inverse_folding=False,
            separate_t=self.cfg.inference.interpolant.codesign_separate_t,
            latents=batch["latents"],
        )
        diffuse_mask = (
            diffuse_mask if diffuse_mask is not None else torch.ones(1, sample_length)
        )
        atom37_traj = [x[0] for x in prot_traj]
        atom37_model_traj = [x[0] for x in model_traj]

        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
        noisy_traj_length = bb_trajs.shape[1]
        assert bb_trajs.shape == (num_batch, noisy_traj_length, sample_length, 37, 3)

        model_trajs = du.to_numpy(torch.stack(atom37_model_traj, dim=0).transpose(0, 1))
        clean_traj_length = model_trajs.shape[1]
        assert model_trajs.shape == (num_batch, clean_traj_length, sample_length, 37, 3)

        aa_traj = [x[1] for x in prot_traj]
        clean_aa_traj = [x[1] for x in model_traj]

        aa_trajs = du.to_numpy(torch.stack(aa_traj, dim=0).transpose(0, 1).long())
        assert aa_trajs.shape == (num_batch, noisy_traj_length, sample_length)

        for i in range(aa_trajs.shape[0]):
            for j in range(aa_trajs.shape[2]):
                if aa_trajs[i, -1, j] == du.MASK_TOKEN_INDEX:
                    print("WARNING mask in predicted AA")
                    aa_trajs[i, -1, j] = 0
        clean_aa_trajs = du.to_numpy(
            torch.stack(clean_aa_traj, dim=0).transpose(0, 1).long()
        )
        assert clean_aa_trajs.shape == (num_batch, clean_traj_length, sample_length)

        return bb_trajs[:, -1], clean_aa_trajs[:, -1]
