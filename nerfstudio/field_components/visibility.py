import torch
import tinycudann as tcnn
import numpy as np
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.ray_samplers import RaySamples
from nerfstudio.fields.base_field import shift_directions_for_tcnn


class Visibility(torch.nn.Module):
    def __init__(
            self,
            aabb,
            spatial_distortion,
            num_levels: int = 8,
            base_res: int = 16,
            max_res: int = 512,
            log2_hashmap_size: int = 17,
            n_neurons: int = 16,
            n_layers: int = 3,
    ):
        super().__init__()

        self.aabb = aabb
        self.spatial_distortion = spatial_distortion

        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.visibility_network = tcnn.NetworkWithInputEncoding(
            n_input_dims=6,
            n_output_dims=1,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "HashGrid",
                        "n_levels": num_levels,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": log2_hashmap_size,
                        "base_resolution": base_res,
                        "per_level_scale": growth_factor,
                    },
                    {
                        "otype": "SphericalHarmonics",
                        "degree": 4
                    },
                ]
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": n_layers - 1,
            },
        )

    def forward(self, ray_samples: RaySamples):
        # get normalized positions
        ## normalize position
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions_flat = positions.view(-1, 3)

        # get normalized directions
        directions = shift_directions_for_tcnn(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)

        # concat
        visibility_network_input = torch.concat([positions_flat, directions_flat], dim=-1)

        # query visibility network
        visibility_network_output = self.visibility_network(visibility_network_input)

        # apply activation function
        pred_transmittance = trunc_exp(visibility_network_output)

        # reshape
        outputs_shape = ray_samples.frustums.directions.shape[:-1]
        pred_transmittance = pred_transmittance.view(*outputs_shape, -1).to(directions)

        return pred_transmittance
