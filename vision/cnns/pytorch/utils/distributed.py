# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
import popdist
import popdist.poptorch
import torch
import horovod.torch as hvd


def handle_distributed_settings(args):
    # Initialise popdist
    if popdist.isPopdistEnvSet():
        init_popdist(args)
    else:
        args.use_popdist = False


def init_popdist(args):
    popdist.init()
    hvd.init()
    args.use_popdist = True
    if popdist.getNumTotalReplicas() != args.replicas:
        logging.warn(f"The number of replicas is overridden by poprun. The new value is {popdist.getNumTotalReplicas()}.")
    args.replicas = int(popdist.getNumLocalReplicas())
    args.popdist_rank = popdist.getInstanceIndex()
    args.popdist_size = popdist.getNumInstances()
    args.popdist_local_rank = hvd.local_rank()


def synchronize_throughput_values(elapsed_time, sample_size):
    elapsed_time = torch.tensor([elapsed_time])
    elapsed_time = torch.max(hvd.allgather(elapsed_time)).item()
    sample_size = hvd.allreduce(torch.tensor(sample_size), op=hvd.Sum).item()
    return elapsed_time, sample_size


def synchronize_latency_values(min_latency, max_latency, avg_latencies):
    min_latency = torch.min(hvd.allgather(torch.tensor([min_latency]))).item()
    max_latency = torch.max(hvd.allgather(torch.tensor([max_latency]))).item()
    avg_latencies = [x.item() for x in allreduce_values(avg_latencies, op=hvd.Average)]
    return min_latency, max_latency, avg_latencies


def allreduce_values(values, op):
    values = [t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in values]
    return hvd.grouped_allreduce(values, op=op)
