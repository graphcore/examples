# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import json
import yaml
import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.python import ipu
from ckpt_utils import parse_h5, get_h5_mapper
from utils import create_ipu_config
from tests.test_utils import check_tensor_relative, getTensorRelativError


def setup_random_seed():
    seed = 1989
    np.random.seed(seed)
    tf.random.set_seed(seed)
    ipu.utils.reset_ipu_seed(seed)


def create_gpu_model(config):
    from tests.tf2_fastspeech2 import TFFastSpeech2, FastSpeech2Config
    conf = FastSpeech2Config(**config)
    model = TFFastSpeech2(conf)
    model._name = "tf2_fastspeech2"
    return model


def create_ipu_model(config, use_pipeline=False):
    from fastspeech2 import build_pipeline_model, build_model
    if use_pipeline:
        model = ipu.keras.PipelineModel(*build_pipeline_model(config))
    else:
        model = tf.keras.Model(*build_model(config))
    model._name = "FastSpeech2"
    return model


@tf.function(experimental_compile=True)
def inference_step(features, model):
    pred = model(features, training=False)
    return pred


@tf.function(experimental_compile=True)
def training_step(features, model):
    with tf.GradientTape() as tape:
        pred = model(features[:-1], training=True)
        loss = calculate_ipu_loss(pred, features)
        grad_ipu = tape.gradient(loss, model.trainable_weights)
    return pred, loss, grad_ipu


def calculate_gpu_loss(out_gpu, ground_truth):
    mb_gpu, ma_gpu, dur_gpu, f0_gpu, eng_gpu = out_gpu
    _, _, duration_gts, f0_gts, energy_gts, mel_gts = ground_truth
    duration_gts = tf.cast(duration_gts, tf.float32)
    loss_mb = tf.reduce_mean(tf.math.abs(mb_gpu - mel_gts))
    loss_ma = tf.reduce_mean(tf.math.abs(ma_gpu - mel_gts))
    loss_dur = tf.reduce_mean(tf.math.abs(dur_gpu - duration_gts))
    loss_f0 = tf.reduce_mean(tf.math.abs(f0_gpu - f0_gts))
    loss_eng = tf.reduce_mean(tf.math.abs(eng_gpu - energy_gts))
    loss = loss_mb + loss_ma + loss_dur + loss_f0 + loss_eng
    print(f"""[GPU loss={loss}]
    loss_mb={loss_mb}, loss_ma={loss_ma}, loss_dur={loss_dur},
    loss_f0={loss_f0}, loss_eng={loss_eng}
    """)
    return loss


def calculate_ipu_loss(out_ipu, ground_truth):
    mb_ipu, ma_ipu, dur_ipu, f0_ipu, eng_ipu = out_ipu
    _, duration_gts, f0_gts, energy_gts, mel_gts = ground_truth
    duration_gts = tf.cast(duration_gts, tf.float32)
    # remove masks
    print(f"Before: {mb_ipu.shape}")
    mb_ipu = mb_ipu[:, :mel_gts.shape[1], :]
    ma_ipu = ma_ipu[:, :mel_gts.shape[1], :]
    print(f"After: {mb_ipu.shape}")
    loss_mb = tf.reduce_mean(tf.math.abs(mb_ipu - mel_gts))
    loss_ma = tf.reduce_mean(tf.math.abs(ma_ipu - mel_gts))
    loss_dur = tf.reduce_mean(tf.math.abs(dur_ipu - duration_gts))
    loss_f0 = tf.reduce_mean(tf.math.abs(f0_ipu - f0_gts))
    loss_eng = tf.reduce_mean(tf.math.abs(eng_ipu - energy_gts))
    loss = loss_mb + loss_ma + loss_dur + loss_f0 + loss_eng
    print(f"""[IPU loss={loss}]
    loss_mb={loss_mb}, loss_ma={loss_ma}, loss_dur={loss_dur},
    loss_f0={loss_f0}, loss_eng={loss_eng}
    """)
    return loss


@pytest.mark.usefixtures("length_regulator_op")
@pytest.mark.ipus(1)
def test_fastspeech2():
    tf.keras.backend.clear_session()
    setup_random_seed()
    input_ids = tf.convert_to_tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
    speaker_ids = tf.convert_to_tensor([0], tf.int32)
    duration_gts = tf.convert_to_tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
    f0_gts = tf.convert_to_tensor(
        [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], tf.float32
    )
    energy_gts = tf.convert_to_tensor(
        [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], tf.float32
    )
    mel_gts = tf.convert_to_tensor(
        np.random.random(size=(1, 10, 80)), tf.float32)
    inputs = [input_ids, speaker_ids,
              duration_gts, f0_gts, energy_gts, mel_gts]

    test_dir = Path(__file__).parent
    with open(Path(test_dir, "test_configs", "test.yaml"), "r") as f:
        gconf = yaml.load(f, Loader=yaml.Loader)
    with open(Path(test_dir, "test_configs", "test.json"), "r") as f:
        iconf = json.load(f)

    cfg = create_ipu_config(
        available_memory_proportion=iconf["available_memory_proportion"],
        num_required_ipus=1,
        partials_type=iconf["partials_type"],
        fp_exceptions=iconf["fp_exceptions"],
        xla_recompute=iconf["xla_recompute"],
        enable_stochastic_rounding=iconf["stochastic_rounding"])

    base_lr = 0.001
    optimizer1 = tf.keras.optimizers.SGD(base_lr)
    optimizer2 = tf.keras.optimizers.SGD(base_lr)

    # run fwd of gpu model to get weights/outputs/loss
    model_gpu = create_gpu_model(gconf["fastspeech2_params"])
    with tf.GradientTape() as tape:
        out_gpu = model_gpu(
            input_ids=input_ids,
            speaker_ids=speaker_ids,
            duration_gts=duration_gts,
            f0_gts=f0_gts,
            energy_gts=energy_gts)
        loss_gpu = calculate_gpu_loss(out_gpu, inputs)

        gnames = [w.name for w in model_gpu.trainable_weights]

    model_gpu.save_weights(Path(test_dir, "full_model_weights.h5"))
    grad_gpu = tape.gradient(loss_gpu, model_gpu.trainable_weights)
    grad_gpu_dict = dict(zip(gnames, grad_gpu))
    weight_gpu_dict = dict(zip(gnames, model_gpu.trainable_weights))

    reload_weights = parse_h5(Path(test_dir, "full_model_weights.h5"))
    i2g_mapper = get_h5_mapper(iconf)
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        model_ipu = create_ipu_model(iconf)
        _ = strategy.run(inference_step, args=[
                         (input_ids, duration_gts, f0_gts, energy_gts), model_ipu])
        wi_names = [w.name for w in model_ipu.weights]
        inames = [w.name for w in model_ipu.trainable_weights]
        weights_to_restore = []
        for w in model_ipu.weights:
            weights_to_restore.append(reload_weights[i2g_mapper[w.name]])
        assert len(weights_to_restore) == len(reload_weights) == len(wi_names), \
            f"Weights loading failed.Loaded {len(weights_to_restore)}/{len(wi_names)}."
        model_ipu.set_weights(weights_to_restore)

        out_ipu, loss_ipu, grad_ipu = strategy.run(
            training_step,
            args=[(input_ids, duration_gts, f0_gts, energy_gts, mel_gts), model_ipu])

        grad_ipu_dict = dict(zip(inames, grad_ipu))
        weight_ipu_dict = dict(zip(inames, model_ipu.trainable_weights))

    optimizer1.apply_gradients(zip(grad_gpu, model_gpu.trainable_weights))
    optimizer2.apply_gradients(zip(grad_ipu, model_ipu.trainable_weights))

    # Compare the outputs
    for og, oi in zip(out_gpu, out_ipu):
        gt = og.numpy()
        it = oi.numpy()
        print(f"Before: {gt.shape}, {it.shape}")
        if len(it.shape) == 3:
            # continue
            if it.shape[1] != gt.shape[1]:
                it = it[:, :gt.shape[1], :]
        print(f"After: {gt.shape}, {it.shape}")
        print(f"Err: {getTensorRelativError(gt, it)}")
        check_tensor_relative(gt, it, margin=5e-5)

    # compare loss
    check_tensor_relative(loss_gpu, loss_ipu, margin=5e-6)
    # compare gradients
    for k, v in i2g_mapper.items():
        # skip non-trainable weights
        if 'position_embeddings' in k:
            continue
        grad_i = grad_ipu_dict[k]
        grad_g = grad_gpu_dict[v]
        if "mel_before/bias" in k:
            print(f"{grad_g}, {grad_i}")
            continue

        print(f"[Gradients]{k}({grad_g.shape}) <--> {v}({grad_i.shape})")
        if isinstance(grad_g, tf.IndexedSlices) and isinstance(grad_i, tf.IndexedSlices):
            check_tensor_relative(grad_g.values, grad_i.values, margin=5e-5)
            print(f"Err: {getTensorRelativError(grad_g.values, grad_i.values)}")
        else:
            check_tensor_relative(grad_g.numpy(), grad_i.numpy(), margin=5e-5)
            print(
                f"Err: {getTensorRelativError(grad_g.numpy(), grad_i.numpy())}")

    # Compare the weights after gradient update
    for k, v in i2g_mapper.items():
        # skip non-trainable weights
        if 'position_embeddings' in k:
            continue
        wi = weight_ipu_dict[k]
        wg = weight_gpu_dict[v]
        if "mel_before/bias" in k:
            print(f"{wg.numpy()}, {wi.numpy()}")
            continue
        print(f"[Weights]{wg.name}({wg.shape}) <--> {wi.name}({wi.shape})")
        check_tensor_relative(wg.numpy(), wi.numpy(), margin=1e-5)
