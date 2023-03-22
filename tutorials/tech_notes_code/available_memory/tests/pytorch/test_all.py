# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest
from tutorials_tests.testing_util import (
    run_command,
    run_python_script_helper,
    CalledProcessError,
)


working_path = Path(__file__).parent.parent.parent


class TestComplete:
    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_run_demo_fits(self):
        run_command(
            "python pytorch_demo.py --available-memory-proportion 0.35",
            working_path,
            ["bs:", ",amp:", ",mean_throughput:"],
        )

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_run_demo_oom_string(self):
        with pytest.raises(CalledProcessError, match="Out of memory on tile"):
            run_python_script_helper(working_path, "pytorch_demo.py", want_std_err=True)

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_run_demo_sweep(self):
        run_command(
            "python sweep.py --batch-size-min 22 --available-memory-min 0.9",
            working_path,
            ["bs=22,amp=0.9,is_oom=", "bs=24,amp=0.9,is_oom="],
        )
