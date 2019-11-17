# Copyright 2019 Graphcore Ltd.
import pytest
import sys
import tempfile

from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())
from get_weights import get_weights  # noqa


@pytest.mark.parametrize('model_name', ['VGG16', 'VGG19', 'ResNet101'])
def test_unsupported_model(model_name):
    with pytest.raises(ValueError):
        get_weights(tempfile.gettempdir(), model_name)


@pytest.mark.parametrize('model_name', ["MobileNet", "MobileNetV2", "NASNetMobile", "DenseNet121", "ResNet50",
                                        "Xception", "InceptionV3", "GoogleNet", "InceptionV1"])
def test_supported_model(model_name):
    assert Path(get_weights(Path(tempfile.gettempdir()), model_name) + ".data-00000-of-00001").exists()
