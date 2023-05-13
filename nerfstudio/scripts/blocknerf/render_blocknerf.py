#!/usr/bin/env python

import json
import torch
from typing import Any, Dict, List, Optional
from typing_extensions import Literal, assert_never
from pathlib import Path
from dataclasses import dataclass, field
import tyro
from nerfstudio.utils import install_checks
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.scripts.render import get_crop_from_json, get_path_from_json, _render_trajectory_video
from nerfstudio.scripts.blocknerf.blocknerf_config import BlockNeRFConfig


@dataclass
class BlockNeRFRenderTrajectory(BlockNeRFConfig):
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    traj: Literal["spiral", "filename", "interpolate"] = "spiral"
    """Trajectory type to render. Select between spiral-shaped trajectory, trajectory loaded from
    a viewer-generated file and interpolated camera paths from the eval dataset."""
    downscale_factor: int = 1
    """Scaling factor to apply to the camera image resolution."""
    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_path: Path = Path("renders/output.mp4")
    """Name of the output file."""
    seconds: float = 5.0
    """How long the video should be."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    interpolation_steps: int = 10
    """Number of interpolation steps between eval dataset cameras."""
    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval."""

    def main(self) -> None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        """Main function."""
        pipeline = self.config.pipeline.setup(
            device=device_str,
            block_npy=self.block_npy,
            merged_model_checkpoint=self.checkpoint,
        )

        install_checks.check_ffmpeg_installed()

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        seconds = camera_path["seconds"]
        if "camera_type" not in camera_path:
            camera_type = CameraType.PERSPECTIVE
        elif camera_path["camera_type"] == "fisheye":
            camera_type = CameraType.FISHEYE
        elif camera_path["camera_type"] == "equirectangular":
            camera_type = CameraType.EQUIRECTANGULAR
        else:
            camera_type = CameraType.PERSPECTIVE
        crop_data = get_crop_from_json(camera_path)
        camera_path = get_path_from_json(camera_path)

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=crop_data,
            seconds=seconds,
            output_format=self.output_format,
            camera_type=camera_type,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(BlockNeRFRenderTrajectory).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa
