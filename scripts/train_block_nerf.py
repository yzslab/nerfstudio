import os
import datetime
import argparse
import numpy as np
import json
import hashlib
from tqdm import tqdm


def sha256sum(filename):
    with open(filename, "rb") as f:
        b = f.read()  # read entire file as bytes
        readable_hash = hashlib.sha256(b).hexdigest()
        return readable_hash


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="blocknerf")
parser.add_argument("--resume-from", type=str, default=None)
parser.add_argument("--block-npy", type=str, required=True)
parser.add_argument("--block-ids", type=str, nargs="*", default=None, help="Only train specified blocks")
args = parser.parse_args()

base_dir = os.path.dirname(args.block_npy)

# load block information from npy file
blocks = np.load(args.block_npy, allow_pickle=True).item()

# build save path
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
block_configs_save_path = os.path.join(base_dir, "block_configs_{}.json".format(timestamp))

# initialize dictionaries
block_config_paths = {}
block_config_hash = {}
# load from resume path
if args.resume_from is not None:
    with open(args.resume_from, "r") as f:
        block_configs = json.load(f)
        block_config_paths = block_configs["configs"]
        block_config_hash = block_configs["hash"]

# train blocks
with tqdm(blocks.keys()) as t:
    for block_id in t:
        t.set_description(block_id)

        # skip empty block
        if len(blocks[block_id]["images"]) == 0:
            continue

        # only train specified blocks
        if args.block_ids is not None and block_id not in args.block_ids:
            continue

        # build experiment name
        experiment_name = "{}_{}".format(args.name, block_id)

        # build json file path
        block_transform_json_path = os.path.join(base_dir, "block_{}.json".format(block_id))
        # calculate json file hash
        json_sha256sum = sha256sum(block_transform_json_path)

        # check whether block has been trained
        if block_id in block_config_paths and block_id in block_config_hash and json_sha256sum == block_config_hash[
            block_id]:
            print("block {} already trained".format(block_id))
            continue

        # invoke training command
        print("start training block {}".format(block_id))
        return_code = os.system(
            "ns-train nerflab --data '{}' --experiment-name '{}' --timestamp '{}' --vis wandb".format(
                block_transform_json_path,
                experiment_name,
                timestamp,
            ))
        if return_code != 0:
            print("error occurred while training block {}".format(block_id))
            exit(return_code)
        # add training output to dictionary
        block_config_paths[block_id] = os.path.join("outputs", experiment_name, "nerflab", timestamp, "config.yml")
        block_config_hash[block_id] = json_sha256sum

        # save result to block configs
        with open(block_configs_save_path, "w") as f:
            json.dump({
                "configs": block_config_paths,
                "hash": block_config_hash,
            }, f, indent=4)
            print("saved to {}".format(block_configs_save_path))
