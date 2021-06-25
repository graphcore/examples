# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
import popdist
import popdist.poptorch
import horovod.torch as hvd


def handle_distributed_settings(opts):
    # Initialise popdist
    if popdist.isPopdistEnvSet():
        init_popdist(opts)
    else:
        opts.use_popdist = False


def init_popdist(args):
    hvd.init()
    args.use_popdist = True
    if popdist.getNumTotalReplicas() != args.replicas:
        logging.warn(f"The number of replicas is overridden by poprun. The new value is {popdist.getNumTotalReplicas()}.")
    args.replicas = int(popdist.getNumLocalReplicas())
    args.popdist_rank = popdist.getInstanceIndex()
    args.popdist_size = popdist.getNumInstances()
    args.popdist_local_rank = hvd.local_rank()
