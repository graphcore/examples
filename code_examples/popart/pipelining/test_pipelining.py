# Copyright 2019 Graphcore Ltd.

import pipelining
import pytest
import argparse
import json


class TestPipeliningPopART(object):
    """Tests for pipelining popART code example"""

    def pytest_namespace():
        return {'pipelining_cycles': 0, 'no_pipelining_cycles': 0}

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_pipelining_running(self):
        args = argparse.Namespace(
            test=True, export=None, no_pipelining=False, report=False)
        session = pipelining.main(args)

        parsed_report = json.loads(session.getExecutionReport())
        pytest.pipelining_cycles = parsed_report['simulation']['cycles']

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_without_pipelining_running(self):

        args = argparse.Namespace(
            test=True, export=None, no_pipelining=True, report=False)
        session = pipelining.main(args)

        parsed_report = json.loads(session.getExecutionReport())
        pytest.no_pipelining_cycles = parsed_report['simulation']['cycles']

    @pytest.mark.category1
    def test_pipelining_faster_than_without(self):

        assert (pytest.pipelining_cycles < pytest.no_pipelining_cycles)


if __name__ == '__main__':
    pytest.main(args=[__file__])
