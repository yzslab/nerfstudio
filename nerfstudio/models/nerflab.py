import tinycudann as tcnn
import numpy as np
from torch.nn import Parameter

from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.cameras.rays import RayBundle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type
from .nerfacto import NerfactoModelConfig, NerfactoModel
from nerfstudio.field_components.visibility import Visibility
from nerfstudio.field_components.spatial_distortions import SceneContraction


@dataclass
class NerflabModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: NerflabModel)

    use_visibility_network: bool = True
    """Train visibility network"""

    visibility_loss_mult: float = 0.001
    """Visibility loss multiplier"""


class NerflabModel(NerfactoModel):
    config = NerflabModelConfig

    def populate_modules(self):
        super().populate_modules()

        scene_contraction = SceneContraction(order=float("inf"))
        self.visibility_network = Visibility(
            aabb=self.scene_box.aabb,
            spatial_distortion=scene_contraction,
        )

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        # weights/transmittance[ray index][sample point index][0] = value
        weights, transmittance = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY],
                                                         return_transmittance=True)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "transmittance": transmittance,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.use_visibility_network:
            outputs["pred_transmittance"] = self.visibility_network(ray_samples)

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # use NeRF output supervise visibility network
        loss_dict["transmittance_loss"] = self.config.visibility_loss_mult * (
                (outputs["transmittance"].detach() - outputs["pred_transmittance"]) ** 2
        ).mean()

        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        group = super().get_param_groups()
        group["visibility_network"] = list(self.visibility_network.parameters())
        return group
