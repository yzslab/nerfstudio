import os.path

import numpy as np
import json
import time

from pathlib import Path

from typing import Dict, List, Tuple, Type
from dataclasses import dataclass, field

import torch
from torch.nn import Parameter

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model, ModelConfig


@dataclass
class BlocknerfModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: BlocknerfModel)


class BlocknerfModel(Model):
    config = BlocknerfModelConfig

    def __init__(self, config: ModelConfig, scene_box: SceneBox, num_train_data: int, block_npy, block_configs, **kwargs) -> None:
        self.block_meta = block_npy
        self.block_configs = block_configs

        super().__init__(config, scene_box, num_train_data, **kwargs)

    def populate_modules(self):
        super().populate_modules()

        # load block configs
        with open(self.block_configs, "r") as f:
            self.block_configs = json.load(f)["configs"]

        # load block information
        self.block_meta = np.load(self.block_meta, allow_pickle=True).item()
        block_bbox_max = []
        block_bbox_min = []
        block_offsets = []
        block_scales = []
        block_centers = []
        block_ids = []
        for i in self.block_meta:
            if i not in self.block_configs:
                continue
            bounding_box = self.block_meta[i]["bbox"]
            block_bbox_max.append(torch.tensor(bounding_box["max"]))
            block_bbox_min.append(torch.tensor(bounding_box["min"]))
            block_offsets.append(torch.tensor([
                self.block_meta[i]["c"][0],
                self.block_meta[i]["c"][1],
                0.0,
            ]))
            block_scales.append(torch.tensor(1. / (bounding_box["max"][0] - bounding_box["min"][0])))
            block_centers.append(torch.tensor(self.block_meta[i]["c"]))
            block_ids.append(i)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.block_bbox_max = torch.stack(block_bbox_max).to(device)
        self.block_bbox_min = torch.stack(block_bbox_min).to(device)
        self.block_scales = torch.stack(block_scales).to(device)
        self.block_offsets = torch.stack(block_offsets).to(device)
        self.block_centers = torch.stack(block_centers).to(device)
        self.block_ids = block_ids

        # ckpt_path = "/tmp/block_models.pth"
        #
        # if os.path.exists(ckpt_path):
        #     block_models = torch.load(ckpt_path)
        #     block_models.to(device)
        # else:

        # load block models
        block_models = torch.nn.ModuleDict()
        for block_id in self.block_configs:
            pipeline = eval_setup(Path(self.block_configs[block_id]), self.config.eval_num_rays_per_chunk, "inference")[1]
            pipeline.model.config.eval_num_rays_per_chunk = 131072
            block_models[block_id] = pipeline.model

            # torch.save(block_models, ckpt_path)


        self.block_models = block_models

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        pass

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        pass

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[
        Dict[str, float], Dict[str, torch.Tensor]]:
        pass

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle):
        # get the first ray origin as camera origin
        # camera_ray_bundle.origins[..., :2] += torch.tensor([4.5, -3.0]).to(camera_ray_bundle.origins.device)
        origin = camera_ray_bundle.origins[0][0][:2]

        # find blocks
        less_than_max = torch.all(origin < self.block_bbox_max.to(origin.device), dim=-1)
        larger_than_min = torch.all(origin > self.block_bbox_min.to(origin.device), dim=-1)
        suitable = less_than_max * larger_than_min
        in_range_block_index = torch.nonzero(suitable)
        in_range_block_ids = [self.block_ids[int(i)] for i in in_range_block_index]

        if not torch.any(suitable):
            # find the closest block
            distances = torch.norm(origin - self.block_centers, dim=-1)
            suitable_block_index = distances.squeeze(-1).argsort()[:2]
        else:
            # only use the two closest blocks
            block_centers = self.block_centers[in_range_block_index]
            distances = torch.norm(origin - block_centers, dim=-1)
            suitable_block_index = in_range_block_index[distances.squeeze(-1).argsort()[:2]]
        # else:
        #     # find the closest block
        #     block_centers = self.block_centers[suitable_block_index]
        #     distances = torch.norm(origin - block_centers, dim=-1)
        #     suitable_block_index = suitable_block_index[distances.argmin()]

        outputs = {}

        suitable_block_ids = []
        block_mean_transmittance = []
        block_output_rgbs = []
        block_output_weights = []
        selected_block_ids = []
        count = 0
        render_consumed = 0
        # use each suitable block to render
        for i in suitable_block_index:
            block_id = self.block_ids[int(i)]
            suitable_block_ids.append(block_id)
            if block_id in self.block_models:
                model = self.block_models[block_id]
                block_scale = self.block_scales[i]
                ray_bundle_for_block = RayBundle(
                    origins=block_scale * (camera_ray_bundle.origins - self.block_offsets[i].to(camera_ray_bundle.origins.device)),
                    directions=camera_ray_bundle.directions,
                    pixel_area=camera_ray_bundle.pixel_area,
                    camera_indices=camera_ray_bundle.camera_indices,
                    nears=camera_ray_bundle.nears,
                    fars=camera_ray_bundle.fars,
                    metadata=camera_ray_bundle.metadata,
                    times=camera_ray_bundle.times,
                )
                # query transmittance
                transmittance = model.get_transmittance_for_camera_ray_bundle(ray_bundle_for_block)
                mean_transmittance = transmittance.mean()
                # discard block if transmittance is too low
                if mean_transmittance <= 0.05 and suitable_block_index.shape[0] > 1:
                    continue
                block_mean_transmittance.append(mean_transmittance)
                start_at = time.time()
                # render
                block_output = model.get_outputs_for_camera_ray_bundle(ray_bundle_for_block)
                render_consumed = time.time() - start_at
                block_output_rgbs.append(block_output["rgb"])
                # calculate weight by block center distance
                block_output_weights.append(self.distance_weight(origin, self.block_centers[i]))
                selected_block_ids.append(block_id)

                outputs[str(count)] = block_output["rgb"]
                count += 1

        if len(block_output_rgbs) > 0:
            block_mean_transmittance = torch.stack(block_mean_transmittance)
            # _, top2index = torch.topk(block_mean_transmittance, 2, True)
            block_output_rgbs = torch.stack(block_output_rgbs)#[top2index]
            block_output_weights = torch.stack(block_output_weights)#[top2index]
            normalized_weights = block_output_weights / block_output_weights.sum()
            print("xy:{}; block_in_range:{}; \n"
                  "    transmittance: {}; used_blocks:{}; weights:{}; render_consumed: {}".format(
                origin.cpu().numpy(),
                in_range_block_ids,
                block_mean_transmittance.cpu().numpy(),
                selected_block_ids,
                normalized_weights.cpu().numpy(),
                render_consumed,
            ))
            # aggregate all block RGB outputs
            rgb = torch.sum(normalized_weights[:, None, None, None] * block_output_rgbs, dim=0)
        else:
            # no suitable block found
            rgb = 0.5 * torch.ones(list(camera_ray_bundle.shape[:2]) + [3]).to(camera_ray_bundle.origins.device)

        outputs["rgb"] = rgb
        return outputs

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}

    def distance_weight(self, point, centroid, p=1):
        return (torch.linalg.norm(point - centroid)) ** (-p)
