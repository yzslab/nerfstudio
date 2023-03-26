from pathlib import Path
from typing import Dict, List, Tuple

from torch.nn import Parameter

from .base_datamanager import DataManagerConfig, DataManager
from ...cameras.rays import RayBundle

from dataclasses import dataclass, field
from nerfstudio.configs import base_config as cfg
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast

from nerfstudio.data.datasets.blocknerf_dataset import BlockNeRFDataset


@dataclass
class BlockNeRFDatamanagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: BlockNeRFDatamanager)

    camera_optimizer: Any = None


class BlockNeRFDatamanager(DataManager):

    def __init__(self):
        super().__init__()

        self.train_dataset = BlockNeRFDataset()

    def get_datapath(self) -> Optional[Path]:
        return Path("/tmp")

    def forward(self):
        pass

    def setup_train(self):
        pass

    def setup_eval(self):
        pass

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        pass

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        pass

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        pass

    def get_train_rays_per_batch(self) -> int:
        return 20480

    def get_eval_rays_per_batch(self) -> int:
        return 20480

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}
