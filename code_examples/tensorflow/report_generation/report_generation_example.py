# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

"""Simple code example of how to handle profiling reports in Tensorflow

Base script for all tests.
"""
import argparse

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.ipu import utils, scopes

datatype = tf.float16
NUM_UNITS_IN = 128
NUM_UNITS_OUT = 256


def parse_args():
    # Handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-execution', action="store_true",
                        help="Enable profiling of the execution of the model, to get the execution report.")
    parser.add_argument('--json-report', action="store_true",
                        help="Generate reports in json format instead of text.")
    parser.add_argument('--var-init-on-cpu', action="store_true",
                        help="Place the variables initialisation on the CPU.")
    parser.add_argument('--no-var-init-profiling', action="store_true",
                        help="Generate reports for the main model graph only.")
    parser.add_argument('--split-reports', action="store_true",
                        help="Write compile and execution reports in two separate files.")
    args = parser.parse_args()
    return args


def model(feature):
    """Simple example model"""
    with tf.variable_scope('fully_connected', use_resource=True):
        weights = tf.get_variable(
            'weights', [NUM_UNITS_IN, NUM_UNITS_OUT],
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            dtype=datatype)
        biases = tf.get_variable(
            'biases', [NUM_UNITS_OUT],
            initializer=tf.constant_initializer(0.0),
            dtype=datatype)

        return tf.nn.xw_plus_b(feature, weights, biases)


if __name__ == "__main__":
    args = parse_args()

    x = tf.placeholder(datatype, shape=[1, NUM_UNITS_IN])

    with scopes.ipu_scope("/device:IPU:0"):
        logits = model(x)

    if args.var_init_on_cpu:
        utils.move_variable_initialization_to_cpu()

    with tf.device('cpu'):
        # Event trace
        trace = gen_ipu_ops.ipu_event_trace()

    # Create a config with profiling on
    opts = utils.create_ipu_config(profiling=True,
                                   use_poplar_text_report=not args.json_report,
                                   profile_execution=args.profile_execution)
    opts = utils.auto_select_ipus(opts, 1)
    utils.configure_ipu_system(opts)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # The "trace" op constantly profiles everything that happens on the IPU,
        # from the moment it's created.
        # Executing the trace op flushes everything
        # it has recorded up to that point and outputs it.
        # Therefore if the variable initializer graph runs on IPU,
        # we can prevent it from being included
        # in the reports with a "session.run(trace)"
        # after it has been run (line above).
        if args.no_var_init_profiling and not args.var_init_on_cpu:
            session.run(trace)

        # Create dummy data
        training_data = np.zeros([1, NUM_UNITS_IN])
        # Run the main graph
        session.run(logits, feed_dict={x: training_data})
        # Execute the event trace op:
        # the result is a list of trace event serialized protobufs.
        raw_report = session.run(trace)
        # These objects can be converted to strings with utility functions,
        # as shown below.
        ext = ".json" if args.json_report else ".txt"
        if args.split_reports:
            compile_reports = utils.extract_compile_reports(raw_report)
            execution_reports = utils.extract_execute_reports(raw_report)
            # These are lists, as long as the number of graphs profiled,
            # except that the
            # execution_reports list will be empty
            # if execution profiling is not enabled.
            # You could save only the last (i.e. relative to the main graph);
            # in this case we save everything.
            if len(compile_reports) > 0:
                with open("compile" + ext, "w", encoding="utf-8") as f:
                    for report in compile_reports:
                        # Each element of the list is a tuple of 2 elements:
                        # the first is a string representing an
                        # auto-generated name of the xla graph
                        # the second is a string containing the
                        # actual report relative to the graph
                        xla_name, report_string = report
                        f.write(xla_name + "\n")
                        f.write(report_string + "\n")
            if len(execution_reports) > 0:
                with open("execution" + ext, "w", encoding="utf-8") as f:
                    for report in execution_reports:
                        xla_name, report_string = report
                        f.write(xla_name + "\n")
                        f.write(report_string + "\n")
        else:
            report = utils.extract_all_strings_from_event_trace(raw_report)
            with open("report" + ext, "w", encoding="utf-8") as f:
                f.write(report)
