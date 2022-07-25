# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import pytest
import sys
import tempfile

from pathlib import Path
import os

sys.path.append(Path(__file__).parent.parent.as_posix())
from get_weights import get_weights  # noqa


@pytest.mark.parametrize('model_name', ['VGG16', 'VGG19', 'ResNet101'])
def test_unsupported_model(model_name):
    with pytest.raises(ValueError):
        get_weights(tempfile.gettempdir(), model_name.lower(), 'float16')


@pytest.mark.parametrize('model_name', ["MobileNet", "MobileNetV2", "NASNetMobile", "DenseNet121", "ResNet50",
                                        "Xception", "InceptionV3", "GoogleNet", "InceptionV1"])
def test_supported_model(model_name):
    save_dir = Path(tempfile.gettempdir() + f'/{model_name}')
    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)
    prefix_ckpt = model_name.lower() + ".ckpt*"
    save_dir = Path(tempfile.gettempdir() + f'/{model_name}')
    weight_path = Path(get_weights(save_dir, model_name.lower(), 'float16') + ".data-00000-of-00001")
    assert weight_path.exists()
    # Remove all generated files
    for files in Path(save_dir).glob(prefix_ckpt):
        files.unlink()
