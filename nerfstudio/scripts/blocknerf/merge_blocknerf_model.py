from __future__ import annotations

import os.path
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import torch
import tyro
import json
import numpy as np
from tqdm import tqdm
from rich.console import Console

from nerfstudio.configs.base_config import ViewerConfig, PrintableConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.server import viewer_utils

CONSOLE = Console(width=120)


@dataclass
class MergeBlockNeRFModel(PrintableConfig):
    """Load a checkpoint and start the viewer."""

    block_configs: Path
    """Path to config JSON file."""

    output_path: Path
    """Path to output merged model."""

    def main(self):
        # load block configs
        with open(self.block_configs, "r") as f:
            block_configs = json.load(f)["configs"]

        model_init_param_dict = {}
        model_config_dict = {}
        model_dict = torch.nn.ModuleDict()

        with tqdm(block_configs.keys()) as t:
            for block_id in t:
                t.set_description(f"Processing block {block_id}")
                # load model from checkpoint
                config, pipeline, _, _ = eval_setup(
                    Path(block_configs[block_id]),
                    eval_num_rays_per_chunk=1 << 15,
                    test_mode="inference",
                    force_device="cpu",
                )

                # get model information
                model = pipeline.model
                model_init_params = {
                    "scene_box":  model.scene_box,
                    "num_train_data": model.num_train_data,
                    "kwargs": model.kwargs,
                }
                model_config = config.pipeline.model

                # store to dict
                model_init_param_dict[block_id] = model_init_params
                model_config_dict[block_id] = model_config
                model_dict[block_id] = model

        # save all model to single checkpoint
        torch.save({
            "init_param_dict": model_init_param_dict,
            "config_dict": model_config_dict,
            "state_dict": model_dict.state_dict(),
        }, self.output_path)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(MergeBlockNeRFModel).main()


if __name__ == "__main__":
    entrypoint()
