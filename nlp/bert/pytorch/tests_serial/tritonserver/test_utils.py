# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import numpy as np


def log_performance_results(model_name, request_type, data_type, number_of_processes, throughputs, latencies):
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    logging.info(
        "-------------------------------------------------------------------------------------------")
    # Standardised metric reporting
    logging.info(f"{model_name}-{request_type}-{number_of_processes} results:")
    logging.info(f"\n\tdata_type: {data_type}")
    logging.info(
        f"\n\tthroughput: {np.mean(throughputs)} samples/sec (mean) (min: {np.min(throughputs)},"
        f" max: {np.max(throughputs)}, std: {np.std(throughputs)})")
    if number_of_processes > 1:
        logging.info(
            f"\n\tthroughput thread combined: {number_of_processes * np.mean(throughputs)} samples/sec (mean)")
    logging.info(
        f"\n\tlatency: {avg_latency} ms (mean) (min: {min_latency}, max: {max_latency})")


class DataGeneratorWrapper:
    def __init__(self, data_generator, input_names):
        self.data_generator = data_generator
        self.input_names = input_names

    def __iter__(self):
        self.gen_iter = iter(self.data_generator)
        return self

    def __next__(self):
        data_dict = next(self.gen_iter)
        data_list = []
        processed_data_list = []
        for name in self.input_names:
            processed_data_list.append(
                data_dict[name].numpy().astype(np.int32))
            data_list.append(data_dict[name])
        return processed_data_list, data_list
