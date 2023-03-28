#!/usr/bin/env python
"""
Starts viewer in eval mode.
"""
from __future__ import annotations

import os.path
import tempfile
import torch
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import tyro
from rich.console import Console

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.server import viewer_utils
from nerfstudio.pipelines.blocknerf_pipeline import BlockNeRFPipelineConfig, BlockNeRFPipeline
from nerfstudio.data.dataparsers.blocknerf_dataparser import BlocknerfDataParserConfig
from nerfstudio.data.datamanagers.blocknerf_datamanager import BlockNeRFDatamanagerConfig
from nerfstudio.models.blocknerf import BlocknerfModelConfig

CONSOLE = Console(width=120)

@dataclass
class BlockNeRFConfig:
    block_npy: Path
    """Path to block npy file."""

    checkpoint: Path
    """Path to merged model checkpoint."""

    config: TrainerConfig = TrainerConfig(
        method_name="blocknerf",
        steps_per_eval_batch=99999999,
        steps_per_save=99999999,
        max_num_iterations=0,
        mixed_precision=True,
        pipeline=BlockNeRFPipelineConfig(
            datamanager=BlockNeRFDatamanagerConfig(
                camera_optimizer=None,
            ),
            model=BlocknerfModelConfig(eval_num_rays_per_chunk=1 << 16),
        ),
        optimizers={
        },
        viewer=ViewerConfig(
            start_train=False,
            # image_format="png",
        ),
        vis="viewer",
    )