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
from scripts.blocknerf.blocknerf_config import BlockNeRFConfig
from nerfstudio.viewer.server.viewer_state import ViewerState

CONSOLE = Console(width=120)


@dataclass
class RunBlockNeRFViewer(BlockNeRFConfig):

    def main(self) -> None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        """Main function."""
        pipeline = self.config.pipeline.setup(
            device=device_str,
            block_npy=self.block_npy,
            merged_model_checkpoint=self.checkpoint,
        )

        config = self.config
        config.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        num_rays_per_chunk = self.config.pipeline.model.eval_num_rays_per_chunk
        config.viewer.num_rays_per_chunk = num_rays_per_chunk

        self._start_viewer(config, pipeline)

    def _start_viewer(self, config: TrainerConfig, pipeline: Pipeline):
        base_dir = config.get_base_dir()
        viewer_log_path = base_dir / config.viewer.relative_log_filename
        viewer_state = ViewerState(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
        )
        banner_messages = [f"Viewer at: {viewer_state.viewer_url}"]

        # We don't need logging, but writer.GLOBAL_BUFFER needs to be populated
        config.logging.local_writer.enable = False
        writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=banner_messages)

        assert viewer_state and pipeline.datamanager.train_dataset
        viewer_state.init_scene(
            dataset=pipeline.datamanager.train_dataset,
            train_state="completed",
        )

        viewer_state.viser_server.set_training_state("completed")
        viewer_state.update_scene(step=1)
        while True:
            time.sleep(0.01)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RunBlockNeRFViewer).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RunViewer)  # noqa
