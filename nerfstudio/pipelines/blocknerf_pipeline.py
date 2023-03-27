from torch.nn import Parameter

from .base_pipeline import Pipeline
from nerfstudio.configs import base_config as cfg
from dataclasses import dataclass, field
from nerfstudio.models.base_model import Model, ModelConfig
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datamanagers.blocknerf_datamanager import BlockNeRFDatamanagerConfig, BlockNeRFDatamanager
from nerfstudio.models.blocknerf import BlocknerfModelConfig


@dataclass
class BlockNeRFPipelineConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: BlockNeRFPipeline)

    datamanager: DataManagerConfig = BlockNeRFDatamanagerConfig()
    """specifies the datamanager config"""

    model: ModelConfig = BlocknerfModelConfig()
    """specifies the model config"""


class BlockNeRFPipeline(Pipeline):
    def __init__(
            self,
            config: BlockNeRFPipelineConfig,
            device: str,
            block_npy: str,
            merged_model_checkpoint: str
    ) -> None:
        super().__init__()
        self.config = config

        self.datamanager = BlockNeRFDatamanager()

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            block_npy=block_npy,
            merged_model_checkpoint=merged_model_checkpoint
        )
        self.model.to(device)

    def get_eval_image_metrics_and_images(self, step: int):
        pass

    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        pass

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}
