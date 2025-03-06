# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from proteinblobs.parsing import parse_train_args
args = parse_train_args()
from proteinblobs.logger import get_logger
logger = get_logger(__name__)
from proteinblobs.multiflow_wrapper import MultiflowWrapper


from omegaconf import OmegaConf
import torch, wandb, os
from proteinblobs.dataset import SeqCollate
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from proteinblobs.dataset import GenieDBDataset, MultiflowDataset
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import hashlib
import numpy as np

torch.set_float32_matmul_precision('medium')

torch.manual_seed(args.seed)
np.random.seed(args.seed)

@rank_zero_only
def init_wandb():
    wandb.init(
        entity="no-graining-mit",
        settings=wandb.Settings(start_method="fork"),
        project="proteinblobs",
        name=args.run_name,
        id=hashlib.md5(str(args).encode("utf-8")).hexdigest(),
        resume='allow', # https://docs.wandb.ai/ref/python/init ----> "allow": if id is set with init(id="UNIQUE_ID") or WANDB_RUN_ID="UNIQUE_ID" and it is identical to a previous run, wandb will automatically resume the run with that id.
        config=args,
    )

if args.wandb:
    init_wandb()

args.__dict__.update({
    'multiflow_yaml': "multiflow_config.yaml"
})
cfg = OmegaConf.load(args.multiflow_yaml)

val_len = 1000
if args.dataset == 'genie':
    dataset = GenieDBDataset(args)
    trainset, valset = torch.utils.data.random_split(dataset, [len(dataset) - val_len, val_len])
elif args.dataset == 'multiflow':
    trainset = MultiflowDataset(args, cfg.pdb_dataset)
    valset = MultiflowDataset(args, cfg.pdb_post2021_dataset)
trainset[0]
valset[0]
print('len trainset', len(trainset))
print('len valset', len(valset))



train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=SeqCollate(args),
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.batch_size,
    collate_fn=SeqCollate(args),
    num_workers=args.num_workers,
    shuffle=False
)

model = MultiflowWrapper(args, cfg)

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    strategy='ddp',
    max_epochs=args.epochs,
    limit_train_batches=args.train_batches or 1.0,
    limit_val_batches=0.0 if args.no_validate else (args.val_batches or 1.0),
    num_sanity_val_steps=0,
    enable_progress_bar=not args.wandb,
    gradient_clip_val=args.grad_clip,
    default_root_dir=os.environ["MODEL_DIR"], 
    callbacks=[
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"], 
            save_top_k=-1,
            every_n_epochs=args.ckpt_freq,
        ),
        ModelSummary(max_depth=2),
    ],
    accumulate_grad_batches=args.accumulate_grad,
    val_check_interval=args.val_freq,
    check_val_every_n_epoch=args.val_epoch_freq,
    logger=False
)


if args.ckpt is not None:
    # if there is an hpc checkpoint in the current workdir, then we use that to resume instead of the file in the args.ckpt
    if any(['hpc' in file for file in os.listdir(os.environ['MODEL_DIR'])]):
        ckpt_path = 'hpc'
    else:
        ckpt_path = args.ckpt
else: 
    ckpt_path = None

if args.validate:
    trainer.validate(model, val_loader, ckpt_path=ckpt_path)
else:
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
