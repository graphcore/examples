# Copyright (c) 2019 Graphcore Ltd. All rights reserved.


import pipelining
import pytest
import argparse
import json


class TestPipeliningPopART(object):
    """Tests for pipelining popART code example"""

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_pipelining_running(self):
        args = argparse.Namespace(
            test=True, export=None, no_pipelining=False, report=False)
        session = pipelining.main(args)

        parsed_report = json.loads(session.getExecutionReport())
        cycles = parsed_report['simulation']['cycles']
        print(f"\nRunning the model with pipelining took {cycles} cycles.")

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_without_pipelining_running(self):

        args = argparse.Namespace(
            test=True, export=None, no_pipelining=True, report=False)
        session = pipelining.main(args)

        parsed_report = json.loads(session.getExecutionReport())
        cycles = parsed_report['simulation']['cycles']
        print(f"\nRunning the model without pipelining took {cycles} cycles.")


if __name__ == '__main__':
    pytest.main(args=[__file__])
