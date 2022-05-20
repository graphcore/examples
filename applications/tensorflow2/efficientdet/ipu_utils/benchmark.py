# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from typing import Callable
from typing import Iterable as T_Iterable
from typing import List, Optional, Union

from ipu_utils import safe_mean


class BenchmarkResult:
    stats_fns = [safe_mean, min, max]

    def __init__(self):
        self._throughputs: List[float] = []
        self._latencies: List[float] = []
        self._total_time: float = None

    def add_step(self, throughput: float, latency: float):
        self._throughputs.append(throughput)
        self._latencies.append(latency)

    def set_total_time(self, time_ms: float):
        self._total_time = time_ms

    @property
    def total_time(self):
        return self._total_time

    def get_stats(self, fns: Optional[Union[T_Iterable[Callable], Callable]] = None) \
            -> Union[List[float], float]:
        if fns is None:
            fns = BenchmarkResult.stats_fns
        elif not isinstance(fns, Iterable):
            return fns(self._throughputs), fns(self._latencies)

        tput_stats = [f(self._throughputs) for f in fns]
        latency_stats = [f(self._latencies) for f in fns]
        return tput_stats, latency_stats

    @classmethod
    def print_report(cls, results: List):
        tput_stats, latency_stats = list(
            zip(*[r.get_stats() for r in results]))

        tput_summary = [f([t[i] for t in tput_stats])
                        for i, f in enumerate(cls.stats_fns)]
        latency_summary = [f([l[i] for l in latency_stats])
                           for i, f in enumerate(cls.stats_fns)]

        all_batch_time_stats = [f([r.total_time for r in results])
                                for f in cls.stats_fns]

        print("Benchmark complete.")
        print("Statistics:")
        if len(results) > 1:
            print(f"\tStats captured over {len(results)} runs.")

        print(f"\tThroughput:        mean = {tput_summary[0]:.2f}, "
              f"min = {tput_summary[1]:.2f}, max = {tput_summary[2]:.2f}")
        print(f"\tPer-batch Latency: mean = {latency_summary[0]:.2f}ms, "
              f"min = {latency_summary[1]:.2f}ms, max = {latency_summary[2]:.2f}ms")
        print(f"\tAll-batch Latency: mean = {all_batch_time_stats[0]:.2f}ms, "
              f"min = {all_batch_time_stats[1]:.2f}ms, max = {all_batch_time_stats[2]:.2f}ms")

    def __repr__(self):
        return f"BenchMarkresult: {self._throughputs}, {self._latencies}"
