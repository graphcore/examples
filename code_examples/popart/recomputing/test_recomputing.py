# Copyright (c) 2019 Graphcore Ltd. All rights reserved.


import recomputing
import pytest
import argparse
import json


class TestRecomputingPopART(object):
    """Tests for recomputing popART code example"""

    def pytest_namespace():
        return {'recomputing_memory': 0, 'no_recomputing_memory': 0}

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_no_recomputing(self):
        args = argparse.Namespace(
            test=True, export=None, report=False, recomputing='OFF')
        session = recomputing.main(args)

        graph_report = json.loads(session.getGraphReport())
        pytest.no_recomputing_memory = sum(
            graph_report['memory']['byTile']['total'])

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_manual_recomputing(self):
        args = argparse.Namespace(
            test=True, export=None, report=False, recomputing='ON')
        session = recomputing.main(args)

        graph_report = json.loads(session.getGraphReport())
        pytest.recomputing_memory = sum(
            graph_report['memory']['byTile']['total'])

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_manual_recomputing_use_less_memory(self):
        print("\n")
        print("Memory use (recomputing) -->", pytest.recomputing_memory)
        print("Memory use (no recomputing) -->", pytest.no_recomputing_memory)
        assert (pytest.recomputing_memory < pytest.no_recomputing_memory)

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_auto_recomputing(self):
        args = argparse.Namespace(
            test=True, export=None, report=False, recomputing='AUTO')
        session = recomputing.main(args)

        graph_report = json.loads(session.getGraphReport())
        mem = sum(
            graph_report['memory']['byTile']['total'])
        print("Memory use (auto recomputing) -->", mem)


if __name__ == '__main__':
    pytest.main(args=[__file__, '-s'])
