from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class BlocknerfDataParserConfig(DataParserConfig):
    """Nerflab dataset config"""

    _target: Type = field(default_factory=lambda: Blocknerf)
    """target class to instantiate"""
    data: Path = Path("data/nerflab/demo")
    """Directory or explicit json file path specifying location of data."""


@dataclass
class Blocknerf(DataParser):
    """Nerfstudio DatasetParser"""

    config: BlocknerfDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        dataparser_outputs = DataparserOutputs(
            image_filenames=[],
            cameras=Cameras(
                fx=800.,
                fy=800.,
                cx=400.,
                cy=400.,
                distortion_params=camera_utils.get_distortion_params(0., 0., 0., 0.),
                height=torch.tensor(800),
                width=torch.tensor(800),
                camera_to_worlds=torch.tensor([]),
                camera_type=CameraType.PERSPECTIVE,
            ),
        )
        return dataparser_outputs
