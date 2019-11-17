# Copyright 2019 Graphcore Ltd.
import subprocess
from pathlib import Path
import pytest
import ctypes


custom_ops_loaded = False


def load_custom_ops():
    bert_dir = Path(__file__).parent.parent.resolve()
    so_path = str(bert_dir / "custom_ops.so")
    global custom_ops_loaded
    if not custom_ops_loaded:
        print("Building Custom Ops")
        subprocess.run(['make'], shell=True, cwd=str(bert_dir))
        ctypes.cdll.LoadLibrary(so_path)
        custom_ops_loaded = True
    return so_path


@pytest.fixture
def custom_ops():
    return load_custom_ops()
