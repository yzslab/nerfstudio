import os
import datetime
import argparse
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="blocknerf")
parser.add_argument("--block-npy", type=str, required=True)
args = parser.parse_args()

blocks = np.load(args.block_npy, allow_pickle=True).item()

base_dir = os.path.dirname(args.block_npy)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

block_configs = {}

for block_id in blocks:
    if len(blocks[block_id]["images"]) == 0:
        continue
    print(block_id)

    experiment_name = "{}_{}".format(args.name, block_id)

    return_code = os.system("ns-train nerflab --data '{}' --experiment-name '{}' --timestamp '{}' --vis wandb".format(
        os.path.join(base_dir, "block_{}.json".format(block_id)),
        experiment_name,
        timestamp,
    ))
    if return_code != 0:
        print("error occurred while training block {}".format(block_id))
        exit(return_code)
    block_configs[block_id] = os.path.join("outputs", experiment_name, "nerflab", timestamp, "config.yml")

with open(os.path.join(base_dir, "block_configs_{}.json".format(timestamp)), "w") as f:
    json.dump(block_configs, f)
