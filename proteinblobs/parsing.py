# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from argparse import ArgumentParser
import subprocess, os, sys


def parse_train_args(args=sys.argv):
    parser = ArgumentParser()

    ## Trainer settings
    
    ## Trainer settings
    group = parser.add_argument_group("Epoch settings")
    group.add_argument("--ckpt", type=str, default=None)
    group.add_argument("--validate", action='store_true', default=False)
    group.add_argument("--num_workers", type=int, default=10)
    group.add_argument("--epochs", type=int, default=100)
    group.add_argument("--train_batches", type=int, default=None)
    group.add_argument("--val_batches", type=int, default=None)
    group.add_argument("--batch_size", type=int, default=16)
    group.add_argument("--val_freq", type=int, default=None)
    group.add_argument("--val_epoch_freq", type=int, default=1)
    group.add_argument("--seed", type=int, default=137)
    group.add_argument("--no_validate", action='store_true')

    ## Inference settings
    group = parser.add_argument_group("Inference Settings")
    group.add_argument("--inference_gating", type=float, default=1)
    group.add_argument("--guidance_weight", type=float, default=1)
    group.add_argument("--seq_guidance_weight", type=float, default=1)
    group.add_argument("--guidance", action='store_true')
    group.add_argument("--sc_guidance_mode", choices=['separate', 'guided', 'unguided'], default='separate', help='which model output to use as self conditinoing input to the models')
    group.add_argument("--interpolate_sc", action='store_true')
    group.add_argument("--only_save_blobs", action='store_true')
    group.add_argument("--interpolate_sc_weight", type=float, default=1)
    group.add_argument("--inference_rot_scaling", type=float, default=10)
    group.add_argument("--fixed_inference_size", type=int, default=None)

    ## Model settings
    group = parser.add_argument_group("Model Settings")
    group.add_argument("--pretrained_mf_path", type=str, default='weights/last.ckpt')
    group.add_argument("--finetune", action='store_true')
    group.add_argument("--freeze_weights", action='store_true')
    group.add_argument("--extra_attn_layer", action='store_true')
    group.add_argument("--blob_attention", action='store_true')
    group.add_argument("--blob_drop_prob", type=float, default=0.0)
    group.add_argument("--self_condition", action='store_true')

    ## Data settings
    group = parser.add_argument_group("Data Settings")
    group.add_argument('--dataset', choices=['multiflow', 'genie'], default='genie')
    group.add_argument("--scop_dir", type=str, default='')
    group.add_argument("--genie_db_path", type=str, default='', help= '')
    group.add_argument("--scop_nmr_csv", type=str, default='scop_nmr_files.csv')
    group.add_argument('--pkl_dir', type=str, default='data')
    group.add_argument('--clustering_dir', type=str, default='proteinblobs/clustering_20rpc')
    group.add_argument('--latents_path', type=str, default='projects/protfid/cache/encodings/SimCLR-ca-dim_32/mf_all.pth')
    group.add_argument('--latents_order', type=str, default='projects/protfid/cache/encodings/SimCLR-ca-dim_32/mf_all_names.pickle')
    group.add_argument("--use_latents", action='store_true')
    group.add_argument('--length_dist_npz', type=str, default=None)
    group.add_argument("--crop", type=int, default=256)
    group.add_argument("--no_crop", action='store_true')
    group.add_argument("--res_per_cluster", type=int, default=20)
    group.add_argument("--min_blob_thresh", type=float, default=5)
    group.add_argument("--max_blob_thresh", type=float, default=10)
    group.add_argument("--dataset_multiplicity", type=int, default=None)
    group.add_argument("--repeat_dataset", type=int, default=1)
    group.add_argument("--no_pad", action='store_true')
    group.add_argument("--overfit", action='store_true')
    group.add_argument("--overfit_rot", action='store_true')

    ## Synthetic Blob data settings
    group = parser.add_argument_group("Data Settings")
    group.add_argument("--synthetic_blobs", action='store_true')
    group.add_argument('--num_blobs', type=int, default=5)
    group.add_argument('--nu', type=int, default=10) # vary the blob anisotropy [10, 20, 50, 100]
    group.add_argument('--psi', type=int, default=5) # FIX this to give proteins with O(200) residues
    group.add_argument('--sigma', type=int, default=8) # vary this compactness parameter
    group.add_argument('--helix_frac', type=float, default=0.5)
    
    ## Logging args
    group = parser.add_argument_group("Logging settings")
    group.add_argument("--print_freq", type=int, default=100)
    group.add_argument("--ckpt_freq", type=int, default=1)
    group.add_argument("--wandb", action="store_true")
    group.add_argument("--save_val", action="store_true")
    group.add_argument("--save_single_pdb", action="store_true")
    group.add_argument("--run_name", type=str, default="default")
    group.add_argument("--inf_batches", type=int, default=4)
    group.add_argument("--designability", action='store_true')
    group.add_argument("--self_consistency", action='store_true')
    group.add_argument("--ref_as_sample", action='store_true')
    group.add_argument("--num_timesteps", type=int, default=500)
    group.add_argument("--designability_freq", type=int, default=1)
    group.add_argument("--num_designability_prots", type=int, default=1000)
    group.add_argument("--pmpnn_path", type=str, default='../ProteinMPNN')

    ## Optimization settings
    group = parser.add_argument_group("Optimization settings")
    group.add_argument("--accumulate_grad", type=int, default=1)
    group.add_argument("--lr_scheduler", action='store_true')
    group.add_argument("--grad_clip", type=float, default=1.)
    group.add_argument("--check_grad", action='store_true')
    group.add_argument('--grad_checkpointing', action='store_true')
    group.add_argument('--adamW', action='store_true')
    group.add_argument('--ema', action='store_true')
    group.add_argument('--ema_decay', type=float, default=0.999)
    group.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)
    os.environ["WANDB_LOGGING"] = str(int(args.wandb))
    return args


