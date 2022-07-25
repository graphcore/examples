# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

from dataset import generate_numpy_data, get_images_labels, tf_fit_dataset, tf_eval_dataset, predict_data_set
from parse_args import parse_params
from unet import unet
from utils import set_seed, pretty_print_nested_list, setup_logger


logger = logging.getLogger('UNet')

if tf.__version__[0] != '2':
    raise ImportError("TensorFlow 2 is required for this example")


def main():
    args = parse_params()
    tf.keras.backend.set_floatx(args.dtype)
    set_seed(args.seed)

    # get data
    if args.host_generated_data:
        X, y, X_test = generate_numpy_data(args)
    else:
        X, y, X_test = get_images_labels(args)

    ds_infer = predict_data_set(args, X_test)
    if args.eval and args.kfold > 1:
        # k fold cross validation
        kfold = KFold(n_splits=args.kfold, shuffle=True)
        # Define per-fold accuracy and loss
        loss_per_fold = []
        acc_per_fold = []
        fold_no = 0

        # Generate indices to split data into training and test set.
        for train, val in kfold.split(X, y):
            logger.info(f"Fold: {fold_no} ........")
            ds_train = tf_fit_dataset(args, X[train], y[train])
            ds_eval = tf_eval_dataset(args, X[val], y[val])
            # cross validation on UNet for each fold
            eval_accuracy, eval_loss = unet(args, ds_train, ds_eval, ds_infer)
            if eval_loss is not None and eval_accuracy is not None:
                loss_per_fold.append(eval_loss)
                acc_per_fold.append(eval_accuracy)

            fold_no += 1

        logger.info(f"{args.kfold}-fold cross validation results:")
        logger.info(
            f"Loss:\n {pretty_print_nested_list(loss_per_fold)}, \n mean:\n {np.mean(np.array(loss_per_fold), axis=0)}.")
        logger.info(
            f"Accuracy:\n {pretty_print_nested_list(acc_per_fold)}, \n mean:\n {np.mean(np.array(acc_per_fold), axis=0)}.")
    else:
        # no cross validation
        ds_train = tf_fit_dataset(args, X[:24], y[:24])
        ds_eval = tf_eval_dataset(args, X[24:], y[24:])

        unet(args, ds_train, ds_eval, ds_infer)


if __name__ == '__main__':
    setup_logger(logging.INFO)
    main()
