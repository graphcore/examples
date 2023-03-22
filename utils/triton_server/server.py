# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from pathlib import Path
import subprocess
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from .utilsTriton import Timeout, GetModelPath


class TritonServer:
    def __init__(self, request, ts_path, model_repo_opt, grpc_port_opt):
        self.model_repo_path = GetModelPath(request.config, model_repo_opt)

        ts_gprc_port = str(request.config.getoption(grpc_port_opt))

        self.docker_name = "triton_at_port_" + ts_gprc_port

        self.ts_proc = subprocess.Popen(
            [
                ts_path + "/tritonserver",
                "--model-repository",
                self.model_repo_path,
                "--backend-directory",
                ts_path,
                grpc_port_opt,
                ts_gprc_port,
            ]
        )

        self.url = "localhost:" + ts_gprc_port
        if self.ts_proc is not None:
            self.wait_until_server_is_started()

    def wait_until_server_is_started(self):
        with Timeout(seconds=120, process=self.ts_proc, error_message="Triton sever start timeout"):
            while True:
                try:
                    with grpcclient.InferenceServerClient(url=self.url) as tsrv:
                        server_ready = tsrv.is_server_ready()
                except InferenceServerException:
                    continue
                else:
                    if server_ready:
                        break

    def __del__(self):
        if self.ts_proc is not None:
            self.ts_proc.kill()

    def terminate(self):
        self.__del__()

    def wait(self, timeout):
        self.ts_proc.wait(timeout)
