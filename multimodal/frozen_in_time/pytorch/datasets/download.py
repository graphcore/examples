# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 Max Bain
# This file has been modified by Graphcore

import argparse
import concurrent.futures
import os

import numpy as np
import requests

import pandas as pd
from tqdm import tqdm


def request_save(url, save_fp):
    try:
        if not (os.path.isfile(save_fp) and os.path.getsize(save_fp) > 100000):
            img_data = requests.get(url, timeout=5).content
            with open(save_fp, 'wb') as handler:
                handler.write(img_data)
                print(f'Video re-downloaded {save_fp}')
    except Exception as e:
        raise ValueError(
            f'Video loading failed for {save_fp}, video loading for this dataset is strict.') from e


def main(args):
    # preproc
    video_dir = os.path.join(args.data_dir, 'videos')
    if not os.path.exists(os.path.join(video_dir, 'videos')):
        os.makedirs(os.path.join(video_dir, 'videos'))

    # ASSUMES THE CSV FILE HAS BEEN SPLIT INTO N PARTS
    partition_dir = args.csv_path.replace('.csv', f'_{args.partitions}')
    # if not, then split in this job.
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)
        full_df = pd.read_csv(args.csv_path)
        df_split = np.array_split(full_df, args.partitions)
        for idx, subdf in enumerate(df_split):
            subdf.to_csv(os.path.join(
                partition_dir, f'{idx}.csv'), index=False)

    relevant_fp = os.path.join(args.data_dir, 'relevant_videos_exists.txt')
    if os.path.isfile(relevant_fp):
        exists = pd.read_csv(os.path.join(
            args.data_dir, 'relevant_videos_exists.txt'), names=['fn'])
    else:
        exists = []

    # ASSUMES THE CSV FILE HAS BEEN SPLIT INTO N PARTS
    # data_dir/results_csvsplit/results_0.csv
    # data_dir/results_csvsplit/results_1.csv
    # ...
    # data_dir/results_csvsplit/results_N.csv

    df = pd.read_csv(os.path.join(partition_dir, f'{args.part}.csv'))

    l = len(df)
    df = df.dropna(axis=0, how='any')
    print(len(df)-l)
    df['rel_fn'] = df.apply(lambda x: os.path.join(x['page_dir'], str(x['videoid'])),
                            axis=1)

    df['rel_fn'] = df['rel_fn'] + '.mp4'

    df = df[~df['rel_fn'].isin(exists)]

    playlists_to_dl = np.sort(df['page_dir'].unique())

    for page_dir in tqdm(playlists_to_dl):
        vid_dir_t = os.path.join(video_dir, page_dir)
        pdf = df[df['page_dir'] == page_dir]
        if len(pdf) > 0:
            if not os.path.exists(vid_dir_t):
                os.makedirs(vid_dir_t)

            urls_todo = []
            save_fps = []

            for idx, row in pdf.iterrows():
                video_fp = os.path.join(
                    vid_dir_t, str(row['videoid']) + '.mp4')

                urls_todo.append(row['contentUrl'])
                save_fps.append(video_fp)

            print(f'Spawning {len(urls_todo)} jobs for page {page_dir}')
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.processes) as executor:
                future_to_url = {executor.submit(
                    request_save, url, fp) for url, fp in zip(urls_todo, save_fps)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Shutter Image/Video Downloader')
    parser.add_argument('--partitions', type=int, default=1,
                        help='Number of partitions to split the dataset into, to run multiple jobs in parallel')
    parser.add_argument('--part', type=int, required=True,
                        help='Partition number to download where 0 <= part < partitions')
    parser.add_argument('--data_dir', type=str, default='data/WebVid',
                        help='Directory where webvid data is stored.')
    parser.add_argument('--csv_path', type=str, default='data/WebVid/release/results_2M_train.csv',
                        help='Path to csv data to download')
    parser.add_argument('--processes', type=int, default=16)
    args = parser.parse_args()

    if args.part >= args.partitions:
        raise ValueError("Part idx must be less than number of partitions")
    main(args)
    # nohup python download.py --part 0 > logs/train_download.log 2>&1 &
