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
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.python import ipu

from spektral.datasets import qm9
from spektral.layers import EdgeConditionedConv, GlobalSumPool
from spektral.utils import label_to_one_hot

from qm9_argparser import get_argparser

################################################################################
# PARAMETERS (defaults set in get_argparser())
################################################################################
parser = get_argparser()
args = parser.parse_args()
gradient_accumulation_count = 6


################################################################################
# CONFIGURE THE DEVICE
################################################################################
cfg = ipu.utils.create_ipu_config(profiling=args.profile,
                                  use_poplar_cbor_report=args.profile,
                                  profile_execution=ipu.utils.ExecutionProfileType.IPU_PROFILE if args.profile else False,
                                  report_directory=args.profile_dir if args.profile else '',
                                  max_report_size=int(5e9))
cfg = ipu.utils.auto_select_ipus(cfg, args.num_ipus)
ipu.utils.configure_ipu_system(cfg)

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

    model = ipu.keras.Model(inputs=[X_in, A_in, E_in],
                            outputs=output,
                            gradient_accumulation_count=1 if args.profile else gradient_accumulation_count)
    optimizer = Adam(lr=args.learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()

    ############################################################################
    # FIT MODEL
    ############################################################################
    tic = time.perf_counter()
    model.fit([X_train, A_train, E_train], y_train, batch_size=args.batch_size, epochs=2 if args.profile else args.epochs, steps_per_epoch=4 if args.profile else None)
    toc = time.perf_counter()
    duration = toc - tic
    print(f"Training time duration {duration}")

    if not args.profile:
        ############################################################################
        # EVALUATE MODEL
        ############################################################################
        print('Testing model')
        tic = time.perf_counter()
        model_loss = model.evaluate([X_test, A_test, E_test],
                                    y_test,
                                    batch_size=args.batch_size)
        print(f"Done. Test loss {model_loss}")

        toc = time.perf_counter()
        duration = toc - tic
        print(f"Testing time duration {duration}")
    print('Completed')
