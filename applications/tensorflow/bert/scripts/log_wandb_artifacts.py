#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
import argparse
import os
import subprocess

import wandb


def get_sdk_version():
    sdk_version = "Unknown"

    sdk_path = os.environ.get("POPLAR_SDK_ENABLED")
    if sdk_path:
        sdk_version = os.path.split(os.path.split(sdk_path)[0])[1]

    return sdk_version


def get_app_version():
    return (subprocess.check_output(["git", "describe", "--always", "--dirty"]).strip().decode())


def parse_cmd_line_arsg():
    def list_str(values):
        return values.split(',')

    parser = argparse.ArgumentParser(description="Upload W&B artifacts to an existing run.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('artifact_dir', type=str, help="Path to the directory to be uploaded.")
    parser.add_argument('--artifact-name',
                        type=str,
                        required=True,
                        help="The artifact name. This will be the same name for multiple versions. For example: bert-large-pretraining-phase1")
    parser.add_argument('--run-id',
                        type=str,
                        required=True,
                        help="W&B run ID (the identifier in the last part of a run URL).")
    parser.add_argument('--project-name', type=str, default='tf-bert', help="The W&B project name.")
    parser.add_argument('--aliases', type=list_str, help="Version aliases (similar to tags), separated by a comma.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_cmd_line_arsg()

    if not os.path.isdir(args.artifact_dir):
        raise ValueError(f"{args.artifact_dir} is not a valid directory.")

    run = wandb.init(project="tf-bert", id=args.run_id, resume="allow")
    artifact_metadata = {
        "public_examples_git_ref": get_app_version(),
        "sdk_version": get_sdk_version(),
    }
    artifact = wandb.Artifact(name=args.artifact_name, type="model", metadata=artifact_metadata)
    artifact.add_dir(args.artifact_dir)
    run.log_artifact(artifact, aliases=args.aliases)
