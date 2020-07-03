"""
Utilty code for constructing optimizers and scheduling learning rates.
"""

# System
import math

# Externals
from tensorflow import keras
# import horovod.tensorflow.keras as hvd
import tensorflow.keras as hvd


def get_optimizer(name, lr, lr_scaling='linear', n_ranks=1,
                  distributed=False, **opt_args):
    """
    Configure the optimizer and scale the learning rate
    """
    # Scale the learning rate
    if lr_scaling == 'linear':
        lr = lr * n_ranks
    elif lr_scaling == 'sqrt':
        lr = lr * math.sqrt(n_ranks)

    # Construct the optimizer
    OptType = getattr(keras.optimizers, name)
    opt = OptType(lr=lr, **opt_args)

    # Distributed optimizer wrapper
    if distributed:
        opt = hvd.DistributedOptimizer(opt)

    return opt
