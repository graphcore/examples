#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
A simple program that uses the PopART library ONNX builder to create
a fully connected layer.
"""
import argparse
import struct
from collections import namedtuple
import numpy as np
import popart
import os
import ctypes


so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "../../custom_ops.so")
ctypes.cdll.LoadLibrary(so_path)

Session = namedtuple('Session', ['session', 'anchors'])
ROWS = 28
COLS = 28
OUTPUT_SIZE = 10

g_sparseMatMulTypeLookup = {
    'DENSE_LHS_SPARSE_RHS_DENSE_OUT': 0,
    'DENSE_LHS_DENSE_RHS_SPARSE_OUT': 1,
    'SPARSE_LHS_SPARSE_RHS_SPARSE_OUT': 2
}


class MNIST_model(object):

    def __init__(self, hidden_size=512):
        self.builder = popart.Builder()
        self.dtype = np.float32
        self.hidden = hidden_size
        self.block_size = [8, 8, 8]
        self.data_type = 'float32'

    def weights_initializer(self, shape_list):
        result = None
        if len(shape_list) == 2:
            result = np.random.normal(0, 1, shape_list) * np.square(2. / shape_list[0])
        if len(shape_list) == 1:
            result = np.zeros(shape_list)
        return result.astype(self.dtype)

    def create_rhs(self, dims, block_size, sparsity_level):
        block_size_row = block_size[0]
        block_size_col = block_size[1]
        num_block_rows = dims[0] // block_size_row
        num_block_cols = dims[1] // block_size_col
        proportion = [sparsity_level, 1 - sparsity_level]
        mask = np.random.choice([0, 1], size=(num_block_rows, num_block_cols), p=proportion)
        while np.sum(mask) == 0:
            mask = np.random.choice([0, 1], size=(num_block_rows, num_block_cols), p=proportion)

        sparse_tensor_shape = [np.sum(mask), block_size[1] * block_size[2]]
        rhs = self.weights_initializer(sparse_tensor_shape)
        nnz_per_2D_slice = [np.sum(mask)]
        # At this point mask is a 2D array, convert it into 1D and return
        return np.array(rhs), nnz_per_2D_slice, mask.flatten()

    def create_proto(self, batch_size):
        input_size = ROWS * COLS
        input_shape = popart.TensorInfo("FLOAT", [batch_size, input_size])
        input_x = self.builder.addInputTensor(input_shape)
        x = input_x
        sparse_mm_type = g_sparseMatMulTypeLookup["DENSE_LHS_SPARSE_RHS_DENSE_OUT"]
        # First matmul
        with self.builder.nameScope('Dense1'):
            b_value = self.weights_initializer([self.hidden])
            b = self.builder.addInitializedInputTensor(b_value)
            rhs_dims_1 = [input_size, self.hidden]
            w_bsr, nnz, sparsity_mask = self.create_rhs(rhs_dims_1, self.block_size, opts.sparsity_level)
            w = self.builder.addInitializedInputTensor(w_bsr)
            transpose_rhs = 0
            x = self.builder.customOp(opName = "BSMatMul",
                                      opVersion=1,
                                      domain = "ai.graphcore",
                                      inputs = [x, w],
                                      attributes = {
                                       "bsr_rhs_lengths_per_2d_plane": nnz,
                                       "matrix_dims": [batch_size, input_size, self.hidden],
                                       "block_size": self.block_size,
                                       "sparsity_mask": sparsity_mask.tolist(),
                                       "bsmatmul_type": sparse_mm_type,
                                       "transpose_rhs": transpose_rhs,
                                       "memory_cycle_ratio": opts.memory_cycle_ratio,
                                       "in_type": self.data_type,
                                       "out_type": self.data_type,
                                       "pp_type": self.data_type
                                      })[0]
            x = self.builder.aiOnnx.add([x, b])
        # Nonlinearity
        x = self.builder.aiOnnx.relu([x])
        # Second matmul
        with self.builder.nameScope('Dense2'):
            output_size = OUTPUT_SIZE
            output_size_padding = (self.block_size[2] - OUTPUT_SIZE % self.block_size[2]) % self.block_size[2]
            if output_size_padding > 0:
                # We might need to pad weight matrix in 2nd dimension to make it divisible by block size
                output_size = output_size + output_size_padding
            lhs_dims_2 = [batch_size, self.hidden]
            rhs_dims_2 = [self.hidden, output_size]
            # We use sparse matmul in the outer layer also,
            # to demonstrate how to use block-sparse API
            # on tensors with sizes not divisible by block size.
            # However, we use sparsity 0 here to get good accuracy numbers
            w2_bsr, w2_nnz, sparsity_mask2 = self.create_rhs(rhs_dims_2, self.block_size, 0.0)

            w2 = self.builder.addInitializedInputTensor(w2_bsr)
            transpose_rhs2 = 0
            x = self.builder.customOp(opName = "BSMatMul",
                                      opVersion = 1,
                                      domain = "ai.graphcore",
                                      inputs = [x, w2],
                                      attributes = {
                                       "bsr_rhs_lengths_per_2d_plane": w2_nnz,
                                       "matrix_dims": [batch_size, self.hidden, output_size],
                                       "block_size": self.block_size,
                                       "sparsity_mask": sparsity_mask2.tolist(),
                                       "bsmatmul_type": sparse_mm_type,
                                       "transpose_rhs": transpose_rhs2,
                                       "memory_cycle_ratio": opts.memory_cycle_ratio,
                                       "in_type": self.data_type,
                                       "out_type": self.data_type,
                                       "pp_type": self.data_type
                                      })[0]
            if output_size_padding > 0:
                # Throw away padding
                axes_value = np.array([0, 1]).astype(np.int32)
                axes = self.builder.aiOnnx.constant(axes_value, "axes")
                start = self.builder.aiOnnx.constant(np.array([0, 0]).astype(np.int32), "start")
                end = self.builder.aiOnnx.constant(np.array([batch_size, OUTPUT_SIZE]).astype(np.int32), "end")
                x = self.builder.aiOnnx.slice([x, start, end, axes])
            b2_value = self.weights_initializer([OUTPUT_SIZE])
            b2 = self.builder.addInitializedInputTensor(b2_value)
            output = self.builder.aiOnnx.add([x, b2])
        # Losses
        self.builder.addOutputTensor(output)
        prob = self.builder.aiOnnx.softmax([output])
        label_shape = popart.TensorInfo("INT32", [batch_size])
        label = self.builder.addInputTensor(label_shape)
        loss = self.builder.aiGraphcore.nllloss([prob, label], debugPrefix = "nllLossVal")
        proto = self.builder.getModelProto()
        return proto, input_x, label, output, loss


def load_mnist(data_dir):
    def _readfile(path):
        with open(path, "rb") as f:
            magic_number, num_items = struct.unpack('>II', f.read(8))
            if magic_number == 2051:
                rows, cols = struct.unpack('>II', f.read(8))
                data = np.fromstring(f.read(), dtype=np.uint8)
                data = data.reshape([num_items, rows * cols])
                data = data.astype(dtype=np.float32)
                data = data / 255.0
            else:
                data = np.fromstring(f.read(), dtype=np.uint8)
                data = data.astype(dtype=np.int32)
            return data
    train_data = _readfile(f"{data_dir}/train-images-idx3-ubyte")
    train_labels = _readfile(f"{data_dir}/train-labels-idx1-ubyte")
    test_data = _readfile(f"{data_dir}/t10k-images-idx3-ubyte")
    test_labels = _readfile(f"{data_dir}/t10k-labels-idx1-ubyte")
    return train_data, train_labels, test_data, test_labels


class DataSet:
    def __init__(self, batch_size, batches_per_step, data, labels):
        self.data = data
        self.labels = labels
        self.num_examples = len(data)
        self.batch_size = batch_size
        self.batches_per_step = min(batches_per_step, self.num_examples // self.batch_size)
        self.inputs_per_step = self.batch_size * self.batches_per_step
        self.steps_per_epoch = self.num_examples // self.inputs_per_step

    def __getitem__(self, key):
        input_begin = key * self.inputs_per_step
        input_end = input_begin + self.inputs_per_step
        data = self.data[input_begin:input_end]
        data = data.reshape([self.batches_per_step, self.batch_size, -1])
        labels = self.labels[input_begin:input_end]
        labels = labels.reshape([self.batches_per_step, self.batch_size])
        return data, labels

    def __iter__(self):
        return (self[j] for j in range(self.steps_per_epoch))

    def __len__(self):
        return self.steps_per_epoch


def get_device(num_ipus, sim=True):
    # Select a device
    deviceManager = popart.DeviceManager()
    if sim:
        options = {
            "compileIPUCode": True,
            "numIPUs": num_ipus,
            "tilesPerIPU": 1216
        }
        device = deviceManager.createIpuModelDevice(options)
    else:
        device = deviceManager.acquireAvailableDevice(num_ipus)
        if device is None:
            print("Failed to acquire IPU. Exiting.")
            quit()
    return device


def init_session(proto, loss, dataFlow, userOpts, device, training=True):
    # Create a session to compile and execute the graph
    if training:
        session = popart.TrainingSession(fnModel=proto,
                                         loss=loss,
                                         deviceInfo=device,
                                         optimizer=popart.ConstSGD(0.001),
                                         dataFlow=dataFlow,
                                         userOptions=userOpts)
    else:
        session = popart.InferenceSession(fnModel=proto,
                                          deviceInfo=device,
                                          dataFlow=dataFlow,
                                          userOptions=userOpts)
    print("Compiling the {} graph.".format("training" if training else "validation"))
    session.prepareDevice()
    session.setRandomSeed(1)

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()
    return Session(session, anchors)


def train(opts):
    if opts.fix_seed:
        print('Fixing the seed for result reproducibility')
        np.random.seed(0)
    train_data, train_labels, test_data, test_labels = load_mnist(opts.data_folder)
    # Limit batches_per_step so the test set isn't evaluated more than once.
    max_value = len(test_data) // opts.batch_size
    if max_value < opts.batches_per_step:
        print("(batches-per-step * batch-size) is larger than test set!\n"
              " Reduced batches-per-step to: {}\n".format(max_value))
        opts.batches_per_step = max_value
    training_set = DataSet(opts.batch_size, opts.batches_per_step, train_data, train_labels)
    test_set = DataSet(opts.batch_size, opts.batches_per_step, test_data, test_labels)
    print("Creating ONNX model.")
    model = MNIST_model(hidden_size=opts.hidden_size)
    proto, data_in, labels_in, output, loss = model.create_proto(opts.batch_size)
    # Describe how to run the model
    anchor_desc = {output: popart.AnchorReturnType("ALL"),
                   loss: popart.AnchorReturnType("ALL")}
    dataFlow = popart.DataFlow(opts.batches_per_step, anchor_desc)
    # Options
    userOpts = popart.SessionOptions()
    # The validation graph by default will be optimized to change all variables to constants
    # This prevents that, which allows for checkpoints to be loaded into the model without recompiling
    userOpts.constantWeights = False
    # Enable auto-sharding
    if opts.num_ipus > 1:
        userOpts.virtualGraphMode = popart.VirtualGraphMode.Auto
    # Enable pipelining
    if opts.pipeline:
        userOpts.enablePipelining = True
    userOpts.separateCallOpPdfs = False
    device = get_device(opts.num_ipus, opts.simulation)
    training = init_session(proto, loss, dataFlow, userOpts, device, training=True)
    validation = init_session(proto, loss, dataFlow, userOpts, device, training=False)
    print("Running training loop.")
    for i in range(opts.epochs):
        # Training
        training.session.weightsFromHost()
        for step, (data, labels) in enumerate(training_set):
            stepio = popart.PyStepIO({data_in: data, labels_in: labels}, training.anchors)
            training.session.run(stepio, 'Epoch ' + str(i) + ' training step' + str(step))
        aggregated_loss = 0
        aggregated_accuracy = 0
        training.session.modelToHost('ckpt.onnx')
        validation.session.resetHostWeights('ckpt.onnx')
        validation.session.weightsFromHost()
        # Evaluation
        for step, (data, labels) in enumerate(test_set):
            stepio = popart.PyStepIO({data_in: data, labels_in: labels}, validation.anchors)
            validation.session.run(stepio, 'Epoch ' + str(i) + ' evaluation step ' + str(step))
            # Loss
            aggregated_loss += np.mean(validation.anchors[loss])
            # Accuracy
            results = np.argmax(validation.anchors[output].reshape([test_set.inputs_per_step, 10]), 1)
            num_correct = np.sum(results == labels.reshape([test_set.inputs_per_step]))
            aggregated_accuracy += num_correct / test_set.inputs_per_step
        # Log statistics
        aggregated_loss /= len(test_set)
        aggregated_accuracy /= len(test_set)
        print("Epoch #{}".format(i + 1))
        print("   Loss={0:.4f}".format(aggregated_loss))
        print("   Accuracy={0:.2f}%".format(aggregated_accuracy * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'Simple MNIST example to test serialized matrix matrix multiplication in PopART',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type = int, default = 32,
                        help = "Set the Batch size")
    parser.add_argument('--batches-per-step', type=int, default=20,
                        help = "Number of minibatches to perform on the Device before returning to the Host."
                        " This will be capped so the Device returns each epoch.")
    parser.add_argument('--epochs', type = int, default = 10,
                        help = "Number of epochs to train for.")
    parser.add_argument('--num-ipus', type = int, default = 1,
                        help = "Number of IPU's")
    parser.add_argument('--pipeline', action = "store_true", default = False,
                        help = "Pipeline the model over IPUs")
    parser.add_argument('--fix-seed', action = "store_true", default = False,
                        help = "Fix the seeds")
    parser.add_argument('--simulation', action = 'store_true',
                        help = "Run the example with an IPU_MODEL device.")
    parser.add_argument('--log-graph-trace', action = 'store_true',
                        help = "Turn on ir logging to display the graph's ops.")
    parser.add_argument('--hidden-size', type = int, default = 400,
                        help = 'The number of neurons in the hidden layer')
    parser.add_argument('--sparsity-level', type = float, default = 0.2,
                        help = 'The level of sparsity (0 = fully dense, 1 = fully sparse)')
    parser.add_argument('--memory-cycle-ratio', type = float, default = 0.0,
                        help='Memory cycle ratio to use in partitioning algorithm')
    parser.add_argument('data_folder', type = str,
                        help = 'Path to mnist data')
    opts = parser.parse_args()
    train(opts)
