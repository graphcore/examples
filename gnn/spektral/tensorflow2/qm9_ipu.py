# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# Copyright (c) 2019 Daniele Grattarola.
# Original code here https://github.com/danielegrattarola/spektral/blob/a2cd265a9440831afc441c1774dd1b7d080a59f8/examples/graph_prediction/qm9_batch.py
"""
IPU Implementation
This example shows how to perform regression of molecular properties with the
QM9 database, using a GNN based on edge-conditioned convolutions in batch mode.
"""
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from spektral.datasets import qm9
from spektral.layers import EdgeConditionedConv, GlobalSumPool
from spektral.utils import label_to_one_hot
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python import ipu
from tensorflow.python.ipu.config import IPUConfig

from qm9_argparser import get_argparser

################################################################################
# PARAMETERS (defaults set in get_argparser())
################################################################################
parser = get_argparser()
args = parser.parse_args()
gradient_accumulation_steps_per_replica, epochs = (1, 2) if args.profile else (6, args.epochs)

################################################################################
# CONFIGURE THE DEVICE
################################################################################
cfg = IPUConfig()
cfg.auto_select_ipus = args.num_ipus
cfg.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
cfg.configure_ipu_system()

# Mixed precision support
tf.keras.backend.set_floatx('float16')

################################################################################
# LOAD DATA
################################################################################
A, X, E, y = qm9.load_data(return_type='numpy',
                           nf_keys='atomic_num',
                           ef_keys='type',
                           self_loops=True,
                           amount=args.amount)  # Set to None to train on whole dataset

y = y[['cv']].values  # Heat capacity at 298.15K

# Preprocessing
X_uniq = np.unique(X)
X_uniq = X_uniq[X_uniq != 0]
E_uniq = np.unique(E)
E_uniq = E_uniq[E_uniq != 0]

X = label_to_one_hot(X, X_uniq)
E = label_to_one_hot(E, E_uniq)

# Parameters
N = X.shape[-2]       # Number of nodes in the graphs
F = X[0].shape[-1]    # Dimension of node features
S = E[0].shape[-1]    # Dimension of edge features
n_out = y.shape[-1]   # Dimension of the target

# Train/test split
data = train_test_split(A, X, E, y, test_size=0.1, random_state=0)

A_train, A_test, X_train, X_test, E_train, E_test, y_train, y_test = [x.astype(np.float32) for x in data]

# global_batch_size = args.micro_batch_size * num_replicas
train_dataset = tf.data.Dataset.from_tensor_slices(((X_train, A_train, E_train), y_train))
train_dataset = train_dataset.batch(args.micro_batch_size, drop_remainder=True)

# Dataset length must be must be divisible by num_replicas, then gradient accumulation count
num_replicas = args.num_ipus
compatible_data_len = len(train_dataset) - len(train_dataset) % (num_replicas * gradient_accumulation_steps_per_replica)
train_dataset = train_dataset.take(compatible_data_len)

test_dataset = tf.data.Dataset.from_tensor_slices(((X_test, A_test, E_test), y_test))
test_dataset = test_dataset.batch(1, drop_remainder=True)

################################################################################
# RUN INSIDE OF A STRATEGY
################################################################################
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
    ############################################################################
    # BUILD MODEL
    ############################################################################
    X_in = Input(shape=(N, F))
    A_in = Input(shape=(N, N))
    E_in = Input(shape=(N, N, S))

    X_1 = EdgeConditionedConv(32, activation='relu')([X_in, A_in, E_in])
    X_2 = EdgeConditionedConv(32, activation='relu')([X_1, A_in, E_in])
    X_3 = GlobalSumPool()(X_2)
    output = Dense(n_out)(X_3)

    model = tf.keras.Model(inputs=[X_in, A_in, E_in],
                           outputs=output)

    model.set_gradient_accumulation_options(
        gradient_accumulation_steps_per_replica=gradient_accumulation_steps_per_replica
    )
    optimizer = Adam(lr=args.learning_rate)

    # `steps_per_execution` is per replica
    steps_per_execution = len(train_dataset) // num_replicas
    model.compile(optimizer=optimizer, loss='mse', steps_per_execution=steps_per_execution)
    model.summary()

    ############################################################################
    # FIT MODEL
    ############################################################################
    train_steps_per_epoch = num_replicas if args.profile else None

    tic = time.perf_counter()
    model.fit(
        train_dataset,
        batch_size=args.micro_batch_size,
        epochs=epochs,
        steps_per_epoch=train_steps_per_epoch
    )
    toc = time.perf_counter()
    duration = toc - tic
    print(f"Training time duration {duration}")

    if not args.profile:
        ############################################################################
        # EVALUATE MODEL
        ############################################################################

        print('Testing model')
        tic = time.perf_counter()
        model_loss = model.evaluate(test_dataset, batch_size=1)
        print(f"Done. Test loss {model_loss}")

        toc = time.perf_counter()
        duration = toc - tic
        print(f"Testing time duration {duration}")
    print('Completed')
