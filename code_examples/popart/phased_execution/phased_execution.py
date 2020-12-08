# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import argparse
import time

import numpy as np

import popart

# The number of IPUs used by this demo
N_IPUS = 2


def parse_args():
    # Handle command line arguments
    parser = argparse.ArgumentParser(description='Phased Execution demo in PopART',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Sets the batch size.')
    parser.add_argument('--dsize', default=250, type=int,
                        help='Size of all the square matrices involved in the computations.')
    parser.add_argument('--num-layers', default=3, type=int,
                        help='Number of layers that constitute the model.')
    parser.add_argument('--batches-per-step', type=int, default=100,
                        help='Number of mini-batches to perform on the device before returning to the host.')
    parser.add_argument('--iters', type=int, default=5,
                        help='Number of iterations to run.')
    parser.add_argument('--dtype', type=str, default="float16",
                        choices=["float32", "float16"],
                        help="Data type for the model.")
    parser.add_argument('--profile', action='store_true',
                        help='Profile the execution and generate a report.')
    parser.add_argument('--profile-dir', type=str, default="./",
                        help="Directory where to save the report files.")
    parser.add_argument('--sharded-execution', action='store_true',
                        help='Run the model by just sharding it over the two devices,'
                        ' without making use of the phased execution.')
    args = parser.parse_args()
    return args


def create_model(phased_execution, batch_size, dsize, num_layers, dtype):
    # Defines a model which is split in several execution phases

    print("Building {} execution model.".format(
        "phased" if phased_execution else "sharded"))
    # Compute the FLOPs/byte loaded from Streaming Memory
    if phased_execution:
        FLOPs = 2 * batch_size * dsize ** 3
        n_bytes = np.dtype(dtype).itemsize * dsize ** 2
        print(
            f"FLOPs / byte loaded for each phase from Streaming Memory: {int(FLOPs / n_bytes)}")

    builder = popart.Builder()
    input_type = "FLOAT16" if dtype == np.float16 else "FLOAT"
    ip = builder.addInputTensor(input_type, [batch_size, dsize, dsize])

    def add_layer(index, in_id):
        # Defines a single layer, consisting of a matmul
        w = builder.addInitializedInputTensor(
            np.random.rand(dsize, dsize).astype(dtype), f"W{index}")
        out = builder.aiOnnx.matmul([in_id, w])
        return out

    out = ip
    for i in range(num_layers):
        # When sharded execution is enabled map the layers on alternate IPUs,
        # to emulate the placement of the phased execution.
        vgid = i % N_IPUS
        # Each layer correspond to a distinct execution phase.
        phase = i
        with builder.executionPhase(phase) if phased_execution else builder.virtualGraph(vgid):
            out = add_layer(i, out)

    last_phase = num_layers - 1
    last_vgid = last_phase % N_IPUS
    with builder.executionPhase(last_phase) if phased_execution else builder.virtualGraph(last_vgid):
        l1 = builder.aiGraphcore.l1loss([out], 0.1)

    builder.addOutputTensor(l1)
    anchor_map = {l1: popart.AnchorReturnType("All")}

    proto = builder.getModelProto()

    return ip, anchor_map, proto


if __name__ == '__main__':
    args = parse_args()

    if args.profile:
        args.iters = 1
        print("Profiling enabled, number of iterations set to one.")

    phased_execution = not args.sharded_execution
    dtype = np.float16 if args.dtype.lower() == "float16" else np.float32

    input_id, anchor_map, proto = create_model(phased_execution=phased_execution,
                                               batch_size=args.batch_size, dsize=args.dsize,
                                               num_layers=args.num_layers, dtype=dtype)

    print("Acquiring device.")
    device = popart.DeviceManager().acquireAvailableDevice(
        numIpus=N_IPUS,
        pattern=popart.SyncPattern.ReplicaAndLadder if phased_execution else popart.SyncPattern.Full)

    opts = popart.SessionOptions()
    if args.profile:
        opts.engineOptions = {"autoReport.all": "true",
                              "autoReport.directory": args.profile_dir}
    if phased_execution:
        # Constant weights cannot be streamed
        opts.constantWeights = False
        opts.executionPhaseSettings.phases = args.num_layers
        opts.executionPhaseSettings.stages = 2
        opts.virtualGraphMode = popart.VirtualGraphMode.ExecutionPhases
        opts.numIOTiles = 128

        varLocation = popart.TensorLocation()
        varLocation.storage = popart.TensorStorage.OffChip
        varLocation.loadTileSet = popart.TileSet.IO
        varLocation.storageTileSet = popart.TileSet.IO
        opts.weightTensorLocationSettings.location = varLocation
    else:
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    print("Compiling.")
    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=popart.DataFlow(
                                          args.batches_per_step, anchor_map),
                                      userOptions=opts,
                                      deviceInfo=device)

    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()

    print("Running.")
    for i in range(args.iters):
        input_data = np.random.rand(args.batches_per_step,
                                    args.batch_size, args.dsize, args.dsize).astype(dtype)
        stepio = popart.PyStepIO({input_id: input_data}, anchors)
        start = time.time()
        session.run(stepio)
        duration = time.time() - start

        print("{0:<8.3} s/iter.\t {1:<8.1f} items/s.".format(
            duration,
            args.batch_size * args.batches_per_step / duration))
