#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import json
import os.path
from dataclasses import dataclass
from pathlib import Path

import tyro
from rich.console import Console

from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = None

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path = eval_setup(self.load_config)

        if self.output_path is None:
            self.output_path = Path(os.path.join("evals", "{}-{}-{}-{}".format(
                config.experiment_name,
                config.method_name,
                config.timestamp,
                os.path.basename(checkpoint_path),
            )))

        self.output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = pipeline.get_average_eval_image_metrics(output_dir=str(self.output_path))
        # self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        (self.output_path / "metrics.json").write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
