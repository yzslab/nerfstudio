# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import Dataset
from .base_dataset import InputDataset

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path


class BlockNeRFDataset(InputDataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    def __init__(self):
        Dataset.__init__(self)

        self.image = torch.ones(800, 800, 3, dtype=torch.float32)
        self.cameras = Cameras(
            fx=1111.,
            fy=1111.,
            cx=400.,
            cy=400.,
            distortion_params=torch.tensor([0., 0., 0., 0., 0.], dtype=torch.float32),
            height=800,
            width=800,
            camera_to_worlds=torch.tensor([
                [
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                ]
            ], dtype=torch.float32),
            camera_type=CameraType.PERSPECTIVE,
        )
        self.metadata = {}

        aabb_scale = 1.
        self.scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

    def __len__(self):
        return 1

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        return 1

    def get_image(self, image_idx: int):
        return self.image

    def get_data(self, image_idx: int) -> Dict:
        return {
            "image_idx": image_idx,
            "image": self.image,
        }

    def get_metadata(self, data: Dict) -> Dict:
        return {}

    @property
    def image_filenames(self) -> List[Path]:
        return [Path("01.png")]
