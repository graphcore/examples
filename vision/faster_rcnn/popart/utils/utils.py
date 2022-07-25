# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import popart
import onnx
from onnx import numpy_helper
import numpy as np


def mappin_txt2dict(txt_path):
    # mapping from torch key to GC key
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    mappin_dic = {}
    for line in lines:
        line = line.strip('\n')
        k, v = line.split()
        mappin_dic[k] = v
    return mappin_dic


def load_from_pth_with_mappin(pth_path, mappin_path):
    mappin_dic = mappin_txt2dict(mappin_path)
    torch_weights = torch.load(pth_path, map_location=torch.device('cpu'))
    initializer = {}
    for torch_key, GC_key in mappin_dic.items():
        initializer[GC_key] = torch_weights[torch_key].detach().numpy().copy()
    return initializer


def is_external_weight(weight):
    return weight.HasField(
        "data_location") and weight.data_location == TensorProto.EXTERNAL


def shape2size(shape):
    result = 1
    for val in shape:
        result = val * result
    return result


def load_onnx(model_path, load_optimizer=False):
    """Load initial weights from an onnx checkpoint.

    Args:
        model_path (str): Path to onnx file.

    Returns:
        Dict: Mapping of popart weight names to numpy values.
    """
    initializers = {}
    # By default onnx.load will look for initializers in the same dir as onnx model.
    # However builder.saveIntializersExternally takes real path or path relative to run dir
    # and stores it in the onnxproto.
    model = onnx.load(model_path, load_external_data=False)

    has_external_data = any(
        is_external_weight(weight) for weight in model.graph.initializer)
    if has_external_data:
        load_external_data_for_model(model, '')

    optimizer_prefix = (popart.reservedAccl1Prefix(),
                        popart.reservedAccl2Prefix(),
                        popart.reservedAcclPrefix(),
                        popart.reservedAccumPrefix(),
                        popart.reservedStepPrefix())

    for weight in model.graph.initializer:
        is_optimizer_state = any(x in weight.name for x in optimizer_prefix)
        if not load_optimizer and is_optimizer_state:
            continue

        if is_external_weight(
                weight) or weight.data_type != onnx.TensorProto.FLOAT16:
            np_weight = numpy_helper.to_array(weight)
        else:
            int_data = np.asarray(weight.int32_data, np.int32)
            np_weight = int_data.view(
                dtype=np.float16).flatten()[:shape2size(weight.dims)].reshape(
                    weight.dims)

        if is_optimizer_state:
            initializers[weight.name] = np_weight.astype(np.float32)
        else:
            initializers[weight.name] = np_weight
    return initializers


def checkNaN_np(np_arr):
    if np_arr.shape == ():
        return np.isnan(np_arr)
    else:
        return True in np.isnan(np_arr)
