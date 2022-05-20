# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import sys
import pytest
import re

run_dir = Path(__file__).parent.parent


class TestBuildAndRun(SubProcessChecker):

    def _run_command(self, args, expected):
        cmd = sys.executable
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        return self.run_command(f"{cmd} {args}",
                                run_dir, expected, env=env)

    @pytest.mark.ipus(1)
    def test_2M_samples(self):
        # Train the NIF model:
        self._run_command(args="train_nif.py --train-samples 2000000 --epochs 200 --layer-count 4 --layer-size 256 --batch-size 1024 --input Mandrill_portrait_2_Berlin_Zoo.jpg --disable-psnr", expected=["Trained NIF."])
        # Reconstruct image using model:
        output = self._run_command(args="predict_nif.py --output result.png --original Mandrill_portrait_2_Berlin_Zoo.jpg", expected = ["Saved image.", r"PSNR:.*$"])
        items = re.findall("PSNR:.*$", output, re.MULTILINE)
        try:
            match = items[-1]
            psnr = float(match.split()[1])
        except:
            print(f"Could not parse PSNR from output:\n{output}")
            psnr = None

        assert psnr > 23.0
        print(f"Test passed (PSRN {psnr})")
