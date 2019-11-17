# Copyright 2019 Graphcore Ltd.
import unittest
import callbacks


class TestPopARTLSTMSyntheticBenchmarks(unittest.TestCase):
    """Tests for the popART LSTM synthetic benchmarks"""

    @classmethod
    def setUpClass(cls):
        pass

    def test_example_runs(self):
        rtts = callbacks.build_and_run_graph(1000)
        print(rtts)
        # Test with a very generous bound on latency (catch only big problems):
        limit = 5e-4
        for k in rtts.keys():
            if rtts[k] > limit:
                raise AssertionError(f"Callback latency {rtts[k]} exceeded limit of {limit} secs for {k}.")
