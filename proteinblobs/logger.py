# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import yaml, logging, socket, os, sys
model_dir = os.environ.get("MODEL_DIR", "./workdir/default")
def get_logger(name):
    logger = logging.Logger(name)
    level = {"crititical": 50, "error": 40, "warning": 30, "info": 20, "debug": 10}[
        os.environ.get("LOGGER_LEVEL", "info")
    ]
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    os.makedirs(model_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(model_dir, "log.out"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        f"%(asctime)s [{socket.gethostname()}:%(process)d] [%(levelname)s] %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

