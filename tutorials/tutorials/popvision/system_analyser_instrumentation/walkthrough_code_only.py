# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# THIS FILE IS AUTOGENERATED. Rerun SST after editing source file: walkthrough.py

import subprocess
import os

mnist_path = "./poptorch_mnist.py"
os.environ["PVTI_OPTIONS"] = '{"enable":"true", "directory": "reports"}'
output = subprocess.run(["python3", mnist_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(output.stdout.decode("utf-8"))

# Generated:2023-05-26T15:09 Source:walkthrough.py SST:0.0.10
