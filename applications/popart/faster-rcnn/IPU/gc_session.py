# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di
import popart
from contextlib import ExitStack
import basic_func as bF
import math
import numpy as np


def add_aux_vars_to_tensor(tensor, prefixes):  # for example grad, accl...
    if hasattr(tensor, 'aux_vars'):
        [tensor.aux_vars.add(_prefix) for _prefix in prefixes]
    else:
        tensor.aux_vars = set()
        [tensor.aux_vars.add(_prefix) for _prefix in prefixes]


# import pdb
class DeviceScope:
    """This class have integrated virtualGraph and pipelineStage together
    """

    # record the number if ipu to used for model
    IPUCount = 0

    def __init__(self, pattern):
        self.builder = bF.get_builder()
        self.pattern = pattern
        self.stack = ExitStack()

    def __enter__(self):
        strs = self.pattern.split("_")
        if len(strs) == 2:
            ipu_id, pipeline_id = strs
        elif len(strs) == 1:
            ipu_id, pipeline_id = strs + ['0']
        else:
            raise RuntimeError('unknown input')
        ipu_id, pipeline_id = int(ipu_id), int(pipeline_id)
        DeviceScope.IPUCount = ipu_id if ipu_id > DeviceScope.IPUCount\
            else DeviceScope.IPUCount

        self.stack.enter_context(self.builder.virtualGraph(int(ipu_id)))
        self.stack.enter_context(self.builder.pipelineStage(int(pipeline_id)))
        return self

    def __exit__(self, *p):
        self.stack.close()
        return False


class TensorCollector:
    def __init__(self, tensors):
        self.tensors = tensors

    def __call__(self, struc):
        if isinstance(struc, bF.TTensor):
            self.tensors.append(struc)


class Assigner:
    def __init__(self, output_dict):
        self.output_dict = output_dict
        self.names = list(output_dict.keys())

    def __call__(self, struc):
        if isinstance(struc, dict):
            for k, v in struc.items():
                if v in self.names:
                    struc[k] = self.output_dict[v]
        elif isinstance(struc, list):
            for idx in range(len(struc)):
                if struc[idx] in self.names:
                    raise NotImplementedError(
                        'list not support inplace change, because it will not be changed in a training(inference) loop, need to work out another solution'
                    )
                    struc[idx] = self.output_dict[struc[idx]]
        else:
            pass


def recursion(complexStruct, func):
    func(complexStruct)
    if isinstance(complexStruct, dict):
        for k in complexStruct:
            recursion(complexStruct[k], func)
    elif isinstance(complexStruct, list):
        for idx in range(len(complexStruct)):
            recursion(complexStruct[idx], func)
    else:
        pass


class Session:
    def __init__(self, outputs, optimizer=None, loss=None):
        self.aux_output_names = [
        ]  # aux var of tensor in popart, like Gradient____xxx, Accl___xxx
        self.outputs = outputs
        self.init_basic(self.outputs)
        print('model is compiling...')
        if optimizer is not None:
            self.init_train_session(optimizer, loss)
        else:
            self.init_inference_session()

    def init_basic(self, outputs):
        self.init_proto()
        self.init_dataFlow(outputs)
        self.init_deviceInfo(bF.get_device_type())
        self.init_options()

    def init_proto(self, ):
        self.proto = bF.get_builder().getModelProto()

    def init_dataFlow(self, complex_outputs):
        self.output_tensors = []
        recursion(complex_outputs, TensorCollector(self.output_tensors))
        anchorTensors = {}
        for t in self.output_tensors:
            anchorTensors[t.getIpuIndex()] = bF.get_anchor_return_type()
            for aux_var in getattr(t, 'aux_vars', []):
                # pdb.set_trace()
                aux_var_name = aux_var + t.getIpuIndex()
                anchorTensors[aux_var_name] = bF.get_anchor_return_type()
                self.aux_output_names.append(aux_var_name)

        self.dataFlow = popart.DataFlow(bF.get_batch_size(), anchorTensors)
        self.batches_per_step = bF.get_batch_size()

    def init_deviceInfo(self, deviceType='ipu'):
        """compute how much ipu should used."""
        if deviceType == 'ipu':
            ipuCount = DeviceScope.IPUCount
            ipu_num = pow(2, math.ceil(math.log2(ipuCount + 1)))
            self.deviceInfo = popart.DeviceManager().acquireAvailableDevice(
                ipu_num)
        elif deviceType == 'cpu':
            self.deviceInfo = popart.DeviceManager().createCpuDevice()
        else:
            raise RuntimeError('unknow device type')

    def init_options(self, ):
        self.options = bF.get_options()

    def init_inference_session(self, ):
        self.session = popart.InferenceSession(fnModel=self.proto,
                                               dataFlow=self.dataFlow,
                                               deviceInfo=self.deviceInfo,
                                               userOptions=self.options)

        self.session.prepareDevice()
        self.session.setRandomSeed(bF.get_seed())
        self.session.weightsFromHost()
        self.anchors = self.session.initAnchorArrays()

    def init_train_session(self, optimizer, loss):
        self.session = popart.TrainingSession(fnModel=self.proto,
                                              loss=loss.getIpuIndex(),
                                              deviceInfo=self.deviceInfo,
                                              optimizer=optimizer.gc_optimizer,
                                              dataFlow=self.dataFlow,
                                              userOptions=self.options)
        self.session.prepareDevice()
        self.session.setRandomSeed(bF.get_seed())
        self.session.weightsFromHost()
        self.anchors = self.session.initAnchorArrays()

    def run(self, feed_dict={}):
        inputs_dict = {}
        for t in feed_dict:
            inputs_dict[t.getIpuIndex()] = np.ascontiguousarray(feed_dict[t])

        if self.batches_per_step > 1:
            raise NotImplementedError
        else:
            stepio = popart.PyStepIO(inputs_dict, self.anchors)

        self.session.run(stepio)
        for t in self.output_tensors:
            t.data = self.anchors[t.getIpuIndex()]
            for aux_output_name in getattr(t, 'aux_vars', []):
                if aux_output_name == 'Gradient___':
                    t.grad = self.anchors[aux_output_name + t.getIpuIndex()]
                elif aux_output_name == 'Accl1___':
                    t.accl1 = self.anchors[aux_output_name + t.getIpuIndex()]
                elif aux_output_name == 'Accl2___':
                    t.accl2 = self.anchors[aux_output_name + t.getIpuIndex()]
                elif aux_output_name == 'Accl___':
                    t.accl = self.anchors[aux_output_name + t.getIpuIndex()]
                else:
                    raise NotImplementedError

        return self.outputs

    def save_model(self, path):
        self.session.modelToHost(path)
