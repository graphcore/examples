# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import time
import copy
import warnings

GLOBAL_V = {
    'builder': popart.Builder(opsets={
        "ai.onnx": 11,
        "ai.graphcore": 1
    }),
    'safe_mode': False,
    'training': False,
    'AnchorReturnType': "FINAL",
    'batchesPerStep': 1,
    'deviceType': 'ipu',
    'options': popart.SessionOptions(),
    'seed': int(time.time()),
    'float_precision': 'FLOAT',
    'available_memory_proportion': None,
    'global_initializer': {},
    'exclude_weights': [],
    'all_weights': [],
    'all_trainable_weights': [],
    'load_strict': False,
}


def set_exclude_weights(exclude_weights):
    if isinstance(exclude_weights, str):
        exclude_weights = [exclude_weights]
    else:
        assert isinstance(exclude_weights, list)
    GLOBAL_V['exclude_weights'] = [ele for ele in exclude_weights]


def get_exclude_weights():
    return GLOBAL_V['exclude_weights']


def get_all_trainable_weights():
    return [*GLOBAL_V['all_trainable_weights']]


def enable_global_initializer(global_initializer):
    warnings.warn('temporary ussage:enable_global_initializer',
                  DeprecationWarning)
    GLOBAL_V['global_initializer'] = copy.deepcopy(global_initializer)


def get_global_initializer():
    return GLOBAL_V['global_initializer']


def set_memory_proportion(proportion):
    GLOBAL_V['available_memory_proportion'] = proportion


def get_memory_proportion():
    return GLOBAL_V['available_memory_proportion']


def get_ai_onnx_version():
    return int(str(get_builder().aiOnnx).replace('AiOnnx', ''))


def set_seed(seed):
    assert isinstance(seed, int)
    GLOBAL_V['seed'] = seed


def get_seed():
    return GLOBAL_V['seed']


def set_options(options, training=True):
    train_options = options.TRAIN
    eval_options = options.EVAL
    common_options = options.COMMON
    _set_options(common_options)
    if training:
        _set_options(train_options)
    else:
        _set_options(eval_options)


def _set_options(options):
    options_cfg = options.pop('CFGS')
    if hasattr(options_cfg, 'VirtualGraphMode'):
        GLOBAL_V['options'].virtualGraphMode = getattr(
            popart.VirtualGraphMode, options_cfg.VirtualGraphMode)
    if hasattr(options_cfg, 'RecomputationType'):
        GLOBAL_V['options'].autoRecomputation = getattr(
            popart.RecomputationType, options_cfg.RecomputationType)
    if hasattr(options_cfg, 'convolutionOptions'):
        GLOBAL_V['options'].convolutionOptions = {}
        for attri, value in options_cfg.convolutionOptions:
            GLOBAL_V['options'].convolutionOptions[attri] = value
    if hasattr(options_cfg, 'accumulationAndReplicationReductionType'):
        GLOBAL_V['options'].accumulationAndReplicationReductionType = getattr(
            popart.ReductionType,
            options_cfg.accumulationAndReplicationReductionType)
    for attr in options:
        if hasattr(GLOBAL_V['options'], attr):
            setattr(GLOBAL_V['options'], attr, options[attr])


def get_options():
    return GLOBAL_V['options']


def set_device(deviceType):
    assert deviceType in ['cpu', 'ipu']
    GLOBAL_V['deviceType'] = deviceType


def get_device_type():
    return GLOBAL_V['deviceType']


def set_batch(N):
    assert isinstance(N, int)
    GLOBAL_V['batchesPerStep'] = N


def get_batch_size():
    return GLOBAL_V['batchesPerStep']


def get_anchor_return_type():
    return popart.AnchorReturnType(GLOBAL_V['AnchorReturnType'])


def train_mode_on():
    assert RuntimeError('Deprecated')
    GLOBAL_V['training'] = True


def train_mode():
    assert RuntimeError('Deprecated')
    return GLOBAL_V['training']


def safe_mode():
    # if safe_mode on, it will check every whether tensor have shape and type or not
    return GLOBAL_V['safe_mode']


def safe_mode_on():
    GLOBAL_V['safe_mode'] = True


def safe_mode_off():
    GLOBAL_V['safe_mode'] = False


def get_builder():
    return GLOBAL_V['builder']


def set_builder(builder):
    # assert isinstance(builder, popart.Builder)
    GLOBAL_V['builder'] = builder


def set_load_strict():
    GLOBAL_V['load_strict'] = True


def load_strict():
    return GLOBAL_V['load_strict']


def load_model(modelPath):
    builder = popart.Builder(modelProtoOrFilename=modelPath,
                             opsets={
                                 "ai.onnx": 11,
                                 "ai.graphcore": 1
                             })
    set_builder(builder)
