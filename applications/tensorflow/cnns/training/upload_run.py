# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Upload training runs to W&B.
"""

import argparse
import json
import pandas as pd
import wandb
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-folder', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--project', default="large-scale-cnn")

    args = parser.parse_args()

    arguments_filename = os.path.join(args.base_folder, 'arguments.json')

    try:
        with open(arguments_filename) as json_data:
            data = json.load(json_data)
    except:
        print("no arguments.json found")
        data = {"params": "empty"}

    wandb.init(
        entity="sw-apps", project=args.project, name=args.name, config=data)

    training_filename = os.path.join(args.base_folder, 'training.csv')
    df1 = pd.read_csv(training_filename, index_col='iteration')
    df1.drop(['step', 'it_time', 'train_acc_batch', 'loss_batch'],
             axis=1, inplace=True)
    rename_columns = {'img_per_sec': 'train_img_per_sec',
                      'lr': 'learning_rate',
                      'train_acc_avg': 'train_accuracy'}

    validation_filename = os.path.join(args.base_folder, 'validation.csv')
    if os.path.exists(validation_filename):
        df2 = pd.read_csv(validation_filename, index_col='iteration')
        df2.drop(['img_per_sec', 'val_time', 'epoch', 'name', 'val_size'],
                 axis=1, inplace=True)

        df = pd.concat([df1, df2], axis=1, join='inner')
        rename_columns['val_acc'] = 'validation_accuracy'
    else:
        df = df1
    df.set_index('epoch', inplace=True)
    df.rename(columns=rename_columns, inplace=True)

    print(df.head())

    for index, row in df.iterrows():
        results = {column_name: column_value
                   for column_name, column_value in row.iteritems()}
        wandb.log(results, step=int(round(index)))

    wandb.save(os.path.join(args.base_folder, "log.txt"))
    wandb.save(os.path.join(args.base_folder, "result*.txt"))
