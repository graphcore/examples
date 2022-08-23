# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import import_helper
from examples_tests.execute_once_per_fs import ExecuteOncePerFS
import os
from pathlib import Path
import pytest
import signal
import socket
import subprocess
import tempfile
from .server import TritonServer

model_repo_opt = "--model-repository"
backend_dir_opt = "--backend-directory"
grpc_port_opt = "--grpc-port"
benchmark_opt = "--benchmark_only"

backend_dir_def = str(Path(__file__).parent.absolute()) + "/backends"


def next_free_port(port=2000, max_port=65535):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return str(port)
        except OSError:
            port += 1
    raise IOError('no free ports')


def pytest_addoption(parser):
    parser.addoption(model_repo_opt, action="store", default=None)
    parser.addoption(backend_dir_opt, action="store", default=backend_dir_def)
    parser.addoption(grpc_port_opt, action="store", default=next_free_port())
    parser.addoption(benchmark_opt, action="store", default=False)


triton_env_lockfile_name = str(
    Path(__file__).parent.absolute()) + "/triton_environment_is_prepared.lock"


@pytest.fixture(scope="session", autouse=True)
@ExecuteOncePerFS(lockfile=triton_env_lockfile_name,
                  file_list=[], timeout=120, retries=20)
def prepare_triton_environment(request):
    poplar_sdk_dir_env = os.getenv('POPLAR_SDK_ENABLED')
    if not poplar_sdk_dir_env:
        pytest.fail("Environment variable POPLAR_SDK_ENABLED is not set!")

    # create triton backend folder - copy files from SDK and arrange it in structure
    # expected by Triton
    poplar_backend_dir = str(
        request.config.getoption(backend_dir_opt)) + "/poplar"
    os.system('mkdir -p ' + poplar_backend_dir)
    os.system('cp -rfL ' + poplar_sdk_dir_env + '/lib ' + poplar_backend_dir)
    os.system('cp -fL ' + poplar_sdk_dir_env + '/../libtriton_poplar.so ' +
              poplar_backend_dir + '/libtriton_poplar.so')

    print("Compiling Triton Server.")
    try:
        ts_dir = tempfile.mkdtemp()
        with open(triton_env_lockfile_name, "w") as telfn:
            telfn.write(str(ts_dir))
        subprocess.run(["sh", "download_and_build_triton_server.sh", str(ts_dir)],
                       cwd=str(Path(__file__).parent.absolute()), check=True,
                       stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        pytest.fail(
            "Failed to download and/or compile Triton Server! stderr: " + str(e.stderr))


@pytest.fixture(scope="module")
def triton_server(request, prepare_triton_environment):
    with open(triton_env_lockfile_name, "r") as telfn:
        ts_exe_path = telfn.read()
    ts_exe_path = ts_exe_path + \
        "/server/mybuild/tritonserver/build/server/mybuild/tritonserver/install/bin/tritonserver"
    if not Path(ts_exe_path).is_file():
        pytest.fail(str(ts_exe_path) + " doesnt exists!")
    server = TritonServer(request, ts_exe_path, model_repo_opt,
                          backend_dir_opt, grpc_port_opt)
    yield server
    server.terminate()


def terminate(signum, frame):
    pytest.exit(False, "Pytest got terminated! signal: ", signum, " frame: ",
                frame)


handled_signals = (signal.SIGABRT, signal.SIGILL, signal.SIGINT,
                   signal.SIGSEGV, signal.SIGTERM)

for sig in handled_signals:
    signal.signal(sig, terminate)
