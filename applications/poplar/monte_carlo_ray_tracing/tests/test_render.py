# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import pytest


root_dir = Path(__file__).parent.parent
build_dir = root_dir.joinpath('test_cmake_build')


@pytest.mark.usefixtures("cmake_build")
class TestBuildAndRun(SubProcessChecker):

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_pathtracer(self):
        cmd = "./ipu_trace -w 480 -h 480 --tile-width 16 --tile-height 12 " \
              "--ipus 1 --refractive-index 1.5 --roulette-depth 5 " \
              "--stop-prob 0.35 --float16-frame-buffer -a 0.003 " \
              "--samples-per-step 100 --samples 1000 --save-interval 10 " \
              "--defer-attach -o test.png"
        self.run_command(cmd, build_dir, [])
        # Check the output image files are valid:
        check_1 = "convert -identify test.png test.png"
        check_2 = "convert -identify test.png.exr test.png.exr"
        self.run_command(check_1, build_dir, ["test.png PNG 480x480"])
        self.run_command(check_2, build_dir, ["test.png.exr EXR 480x480"])

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_compile_and_load(self):
        compile_command = "./ipu_trace -w 640 -h 636 --tile-width 16 --tile-height 12 " \
            "--ipus 2 --samples-per-step 200 --defer-attach --use-simd -o test2.png " \
            "--compile-only --save-exe tracer_graph"
        self.run_command(compile_command, build_dir, [])
        run_command = "./ipu_trace -w 640 -h 636 --tile-width 16 --tile-height 12 " \
            "--ipus 2 --refractive-index 1.5 --roulette-depth 5 " \
            "--stop-prob 0.35 -a 0.003 " \
            "--samples-per-step 200 --samples 400 " \
            "-o test2.png --load-exe tracer_graph"
        self.run_command(run_command, build_dir, [])
        # Check the output image files are valid:
        check_1 = "convert -identify test2.png test2.png"
        check_2 = "convert -identify test2.png.exr test2.png.exr"
        self.run_command(check_1, build_dir, ["test2.png PNG 640x636"])
        self.run_command(check_2, build_dir, ["test2.png.exr EXR 640x636"])
