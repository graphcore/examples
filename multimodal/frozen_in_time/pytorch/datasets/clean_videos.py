# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pandas as pd
import os
import argparse


def clean_videos(args):
    df = pd.read_csv(args.csv_path)

    l = len(df)
    df = df.dropna(axis=0, how="any")
    print(l - len(df))
    df["rel_fn"] = df.apply(lambda x: os.path.join(x["page_dir"], str(x["videoid"])), axis=1)

    df["rel_fn"] = args.video_path + df["rel_fn"] + ".mp4"

    def is_exsist(file_path):
        return os.path.isfile(file_path) and os.path.getsize(file_path) > 100000

    df["ok"] = df.apply(lambda x: is_exsist(x["rel_fn"]), axis=1)
    df_clean = df[df["ok"]].drop(["ok", "rel_fn"], axis=1)
    print(l - len(df_clean))
    df_clean.to_csv(args.clean_csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Cleaner")
    parser.add_argument(
        "--video_path", type=str, default="data/WebVid/videos/", help="Directory where webvid data is stored."
    )
    parser.add_argument(
        "--csv_path", type=str, default="data/WebVid/release/results_2M_train.csv", help="Raw csv data path"
    )
    parser.add_argument(
        "--clean_csv_path",
        type=str,
        default="data/WebVid/metadata/results_2M_training.csv",
        help="New csv data path after cleaned",
    )
    args = parser.parse_args()

    clean_videos(args)
