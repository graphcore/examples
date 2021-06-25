# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import pytest


root_dir = Path(__file__).parent.parent
build_dir = root_dir.joinpath('test_cmake_build')


class TestBuildAndRun(SubProcessChecker):

    def setUp(self):
        build_dir.mkdir(exist_ok=False)
        self.run_command("cmake ..", build_dir, [])
        self.run_command("make", build_dir, [])

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_pathtracer(self):
        cmd = "./ipu_trace -w 480 -h 480 --tile-width 16 --tile-height 12 " \
              "--ipus 1 --refractive-index 1.5 --roulette-depth 5 " \
              "--stop-prob 0.35 --float16-frame-buffer -a 0.003 " \
              "--samples-per-step 100 --samples 1000 --save-interval 10 " \
              "-o test.png"
        self.run_command(cmd, build_dir, [])
        # Check the output image files are valid:
        check_1 = "convert -identify test.png test.png"
        check_2 = "convert -identify test.png.exr test.png.exr"
        self.run_command(check_1, build_dir, ["test.png PNG 480x480"])
        self.run_command(check_2, build_dir, ["test.png.exr EXR 480x480"])
