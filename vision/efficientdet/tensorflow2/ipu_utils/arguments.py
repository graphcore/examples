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

import argparse
import enum
import json
import logging
import os
import re
import sys
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Text,
    Union,
)

import tensorflow as tf
import yaml
from tensorflow.python import ipu


class StoreBoolOverridable(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, default=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, nargs='?', default=default, **kwargs)

        self.original_default = default

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self._parse(values))

    def _parse(self, value: Optional[Union[bool, Text]]) -> bool:
        print(self.dest, value)
        if value is None:
            return not self.original_default
        if isinstance(value, bool):
            return value
        if value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        raise argparse.ArgumentTypeError(
            f'{value} is not a valid boolean value')


class StoreTrueOverridable(StoreBoolOverridable):
    def __init__(self, option_strings, dest, default=False, **kwargs):
        super().__init__(option_strings, dest, default=default, **kwargs)


class StoreFalseOverridable(StoreBoolOverridable):
    def __init__(self, option_strings, dest, default=True, **kwargs):
        super().__init__(option_strings, dest, default=default, **kwargs)


class CustomStringTypeAction(argparse.Action):
    @classmethod
    def to_type(cls, values: Any):
        raise NotImplementedError(
            "This custom action should override the 'to_type' class method to convert to a native type.")


class PrecisionAction(CustomStringTypeAction):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)
        valid_choices = ("uint8", "float", "half")
        are_valid = [c in valid_choices for c in self.choices]
        if not all(are_valid):
            raise NotImplementedError(
                f"Unsupported value(s) provided to choices list: {[valid_choices[i] for i, v in are_valid if not v]}")

    def __call__(self, parser, namespace, values, option_string=None):
        values = PrecisionAction.to_type(values)
        setattr(namespace, self.dest, values)

    @classmethod
    def to_type(cls, values: Text) -> tf.dtypes.DType:
        if values == "float":
            return tf.float32
        elif values == "half":
            return tf.float16
        elif values == "uint8":
            return tf.uint8
        elif values is not None:
            raise NotImplementedError(
                "Precision must be one of (float32, half, uint8)")


class ConnectionTypeAction(CustomStringTypeAction):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        values = ConnectionTypeAction.to_type(values)
        setattr(namespace, self.dest, values)

    @classmethod
    def to_type(cls, values: Text) -> ipu.config.DeviceConnectionType:
        if values == "always":
            return ipu.config.DeviceConnectionType.ALWAYS
        elif values == "never":
            return ipu.config.DeviceConnectionType.NEVER
        elif values == "on_demand":
            return ipu.config.DeviceConnectionType.ON_DEMAND
        raise NotImplementedError(
            "Connection type must be one of (always, never, on_demand).")


class SchedulerSelectionAction(CustomStringTypeAction):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(
            option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        values = SchedulerSelectionAction.to_type(values)
        setattr(namespace, self.dest, values)

    @classmethod
    def to_type(cls, values: Text) -> ipu.config.SchedulingAlgorithm:
        if values == "best":
            return ipu.config.SchedulingAlgorithm.CHOOSE_BEST
        elif values == "clustering":
            return ipu.config.SchedulingAlgorithm.CLUSTERING
        elif values == "post_order":
            return ipu.config.SchedulingAlgorithm.POST_ORDER
        elif values == "look_ahead":
            return ipu.config.SchedulingAlgorithm.LOOK_AHEAD
        elif values == "shortest_path":
            return ipu.config.SchedulingAlgorithm.SHORTEST_PATH
        raise NotImplementedError(
            "Scheduler scheme must be one of (best, clustering, post_order, look_ahead, shortest_path).")


def read_env_engine_opts(args: argparse.Namespace):
    opts = os.environ.get("POPLAR_ENGINE_OPTIONS", None)
    if opts is None:
        return

    opts = json.loads(opts)

    # If the user has set the profile directory, set the internal flag accordingly.
    # If there is a clash between the two, raise an error so there are no surprises.
    if opts.get("autoReport.all", False) == "true" and "autoReport.directory" in opts:
        if args.profile_dir is None:
            args.profile_dir = opts["autoReport.directory"]
        else:
            raise RuntimeError("Profile directory has been set both in the environment and on the command line. "
                               "Please only set one or the other (note that debug.cbor may be incorrectly stored "
                               "located in the application root the command line option")

    if "debug.instrument" in opts:
        env_skip_execution_report = opts["debug.instrument"] != "true"

        # Check the raw command line to check if the execution report was set from there
        if "--skip-execution-report" in sys.argv and args.skip_execution_report != env_skip_execution_report:
            warnings.warn("Execution report request provided on both environment and command line. "
                          "Command line takes precensence, so environment will be ignored.")
        else:
            args.skip_execution_report = env_skip_execution_report

    if "opt.internalExchangeOptimisationTarget" in opts:
        if args.opt_internal_exchange_target is not None:
            warnings.warn("Internal exchange target has been set in both the environment and on the command line."
                          "Command line takes precedence so environment will be ignored.")
        else:
            args.opt_internal_exchange_target = opts["opt.internalExchangeOptimisationTarget"]


def check_config_parameters(parser: argparse.ArgumentParser, yaml_config: Dict[Text, Any]):
    all_dests = []
    for g in parser._action_groups:
        all_dests = all_dests + [a.dest for a in g._actions]

    invalid_config_params = [k for k in yaml_config if k not in all_dests]
    for i in invalid_config_params:
        logging.warning(f"Invalid config parameter detected: {i}")


def apply_custom_actions_to_config(parser: argparse.ArgumentParser, yaml_config: Dict[Text, Any]):
    """ We have a number of custom actions to convert strings to the relevant datatypes. When we load
    a config, it will bring this in as the native strings. We need to manually convert them using the
    custom actions. This runs through all actions in the parser, checks if they're custom, and then
    if they're in the YAML config, it'll convert them using the `to_type` method."""
    for g in parser._action_groups:
        dests = {a.dest: a for a in g._actions if isinstance(
            a, CustomStringTypeAction) and a.dest in yaml_config}
        for d, a in dests.items():
            yaml_config[d] = a.to_type(yaml_config[d])


def parse_args(arg_list: Optional[List[Text]] = None) -> argparse.Namespace:
    pparser = argparse.ArgumentParser(
        "EfficientDet Configuration name", add_help=False)
    pparser.add_argument(
        "--model-name", help="Model name to use - sets up the configuration options.")
    pparser.add_argument("--config", type=str, default="efficientdet", choices=("efficientdet", "efficientdet-low-latency"),
                         help="Which model config to load, EfficientDet for throughput (default) or for single-image low-latency.")
    pargs, remaining_args = pparser.parse_known_args(args=arg_list)
    model_name = pargs.model_name

    if model_name is None and '--help' not in remaining_args:
        pparser.error("Model name must be supplied.")
    elif "--help" not in remaining_args:
        if re.match(r"efficientdet-d[0-4]", model_name) is None:
            pparser.error(
                "Model name must be of the form 'efficientdet-d{0...3}'. D4+ are not currently supported")

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("Input Parameters")
    group.add_argument("--dataset-type", default="generated",
                       choices=("generated", "repeated-image",
                                "image-directory", "single-image"),
                       help="The type of input data to use. If set to 'generated', random uniform data will be "
                            "created on the host. 'repeated-image' mode will fill the batch/dataset with the "
                            "same image. 'single-image' will force batch-size and benchmark-repeats to be 1.")
    group.add_argument("--image-path", default=None,
                       help="Location of the test image(s). If dataset-type is 'single-image' or 'repeated-image', "
                            "this should be the path to a specific image. If dataset-type is 'image-directory', "
                            "this should be a directory containing one or more images.")
    group.add_argument("--image-size", default=None, type=str,
                       help="Image size, either a single number for square images or {width}x{height} otherwise."
                            "It will default to the image size from the paper for a given model size.")
    group.add_argument("--micro-batch-size", type=int,
                       default=1, help="Input batch size")
    group.add_argument("--dataset-prefetch-buffer", type=int, default=0,
                       help="The size fo the TF dataset buffer. Note that this is separate to the "
                       "IPU's prefetch depth which focuses on moving the data closer to the IPU")

    group = parser.add_argument_group("Model Parameters")
    group.add_argument("--model-dir", default=None,
                       help="Location of the checkpoint to run (Defaults to tmp/<model_name>).")
    group.add_argument("--hparams", default="", type=str,
                       help="Comma separated key/value pairs, or a YAML file.")
    group.add_argument("--random-weights", action="store_true",
                       help="If present, the model does not attempt to load real weights, and instead uses random initialisation.")
    group.add_argument("--onchip-nms", action=StoreFalseOverridable,
                       help="Use the IPU custom NMS operator to run post-processing on-chip before streaming back to host.")

    group = parser.add_argument_group("IPU Parameters")
    group.add_argument("--synthetic", action=StoreTrueOverridable,
                       help="Enable on-device synthetic random data (avoids host-stream connections, removes IO)")
    group.add_argument("--partials-precision",
                       choices=("float", "half"),
                       default=None,
                       action=PrecisionAction,
                       help="The precision of the convolution and matmul partials (default: matches the --model-precision)")
    group.add_argument("--model-precision",
                       choices=("float", "half"),
                       default=tf.float16,
                       action=PrecisionAction,
                       help="The precision of the input image after streaming onto the device (default: float)")
    group.add_argument("--io-precision",
                       choices=("uint8", "float", "half"),
                       default=tf.uint8,
                       action=PrecisionAction,
                       help="The input image dtype that is streamed onto the device (default: uint8)")
    group.add_argument("--available-memory-proportion",
                       type=float,
                       default=0.6,
                       help="Set the amount of memory available to the convolution planner. Higher values are faster at the expense of memory.")
    group.add_argument("--ipu-connection-type", choices=("always", "never", "on_demand"), default=ConnectionTypeAction.to_type("always"),
                       action=ConnectionTypeAction,
                       help="When to request a connection to IPUs. Always (default) will connect to an IPU during compilation. on_demand will "
                            "wait until after compilation. Never will not request an IPU connection")

    # These are low-level optimisations, here be dragons...
    group = parser.add_argument_group("IPU Specific Optimisations")
    group.add_argument("--opt-device-rearrange", action=StoreTrueOverridable,
                       help="Explicitly rearrange tensors on the device rather than allowing TF to perform on-host. Potentially increases live memory")
    group.add_argument("--opt-use-io-tiles", action=StoreTrueOverridable,
                       help="Overlap IO with compute to reduce latency")
    group.add_argument("--opt-num-io-tiles", type=int, choices=range(32, 193),
                       help="How many IO tiles to use to parallelise the host->device IO.")
    group.add_argument("--opt-prefetch-data-streams", action=StoreTrueOverridable,
                       help="By default, the prefetching of data for data streams on the host will be not overlapped with execution on the IPU. Set "
                            "this flag to enable it")
    group.add_argument("--opt-prefetch-depth", default=128, type=int,
                       help="The number of elements Poplar will prefetch")
    group.add_argument("--opt-conv-dithering", action=StoreTrueOverridable,
                       help="Enable convolution dithering to spread tensor allocation more evenly over tiles.")
    group.add_argument("--opt-internal-exchange-target", default=None, choices=("memory", "balanced", "cycles"),
                       help="What balance of heuristics to use when generating exchange code. cycles will focus completely on "
                            "speed whereas balanced will sacrifice some speed to attempt to reduce the amount of always live memory produced.")
    group.add_argument("--opt-liveness-scheduler", default=SchedulerSelectionAction.to_type("best"),
                       action=SchedulerSelectionAction,
                       choices=("best", "clustering", "post_order",
                                "look_ahead", "shortest_path"),
                       help="Which scheduler heuristic should be used to reduce liveness.")
    group.add_argument("--opt-scheduler-lookahead-depth", type=int, default=5,
                       help="Only used when scheduler type is look-ahead - The maximum distance to look into the future when considering valid schedules.")
    group.add_argument("--opt-scheduler-lookahead-search-space", type=int, default=64,
                       help="Only used when scheduler type is look-ahead - The maximum number of nodes to consider when building the tree of future schedules.")
    # End optimisations block

    group = parser.add_argument_group("Misc Parameters")
    group.add_argument("--debug", action=StoreTrueOverridable,
                       help="Enable debug info")
    group.add_argument("--log-level", type=str, default="warning",
                       choices=("debug", "info", "warning", "error", "fatal"))
    group.add_argument("--profile-dir", type=str, default=None,
                       help="Path into which to store profile data. If not provided, profiles are not created.")
    group.add_argument("--skip-execution-report", action=StoreTrueOverridable,
                       help="If set, Poplar won't use instrumentation while gathering profile data, so it won't create an execution report.")
    group.add_argument("--num-repeats", type=int, default=20,
                       help="The number of times to repeat the prediction to account for variations in machine load.")
    group.add_argument("--output-predictions", action=StoreTrueOverridable,
                       help="If true, will output the final predictions as annotated image files")
    group.add_argument("--benchmark-host-postprocessing", action=StoreTrueOverridable,
                       help="If provided, the off-chip post-processing is included in the latency measurements. Otherwise only the model time is used.")
    group.add_argument("--output-dir", default="./outputs",
                       help="Directory for annotated output images")
    group.add_argument("--benchmark-repeats", default=100, type=int,
                       help="The number of times to repeat the dataset when running the benchmark. Longer datasets amortise the cost of model IO")

    config_file = os.path.join(os.path.dirname(
        __file__), f"../ipu_configs/{pargs.config}.yml")

    yaml_args = dict()
    if model_name is not None:
        with open(config_file, "r") as f:
            try:
                yaml_config = yaml.safe_load(f)
                if yaml_config is not None and model_name in yaml_config:
                    yaml_args.update(**yaml_config[model_name])
                    check_config_parameters(parser, yaml_args)
                    apply_custom_actions_to_config(parser, yaml_args)
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

    if len(yaml_args) > 0:
        parser.set_defaults(**yaml_args)

    args = parser.parse_args(remaining_args)
    read_env_engine_opts(args)

    args.model_name = model_name

    if args.model_dir is None:
        args.model_dir = f"./tmp/{args.model_name}"

    if args.partials_precision is None:
        args.partials_precision = args.model_precision

    if args.dataset_type != "generated" and args.image_path is None:
        parser.error(
            "If input type is not 'generated', the '--image-path' argument must be set.")
    elif args.dataset_type == "generated" and args.image_path is not None:
        logging.warning(
            "An image path has been provided, but the application has been run with host-generated data. Image path will be ignored.")

    if args.dataset_type == "single-image":
        logging.warning(
            "Running in single-image mode. Overwriting the batch size and benchmark repeats to 1.")
        args.micro_batch_size = 1
        args.benchmark_repeats = 1

    if args.onchip_nms:
        logging.info(
            "Using IPU NMS operator. Host post-processing is disabled.")
        args.benchmark_host_postprocessing = False

    if int(args.model_name[-1]) > 3 and args.opt_internal_exchange_target is None:
        args.opt_internal_exchange_target = "memory"

    if args.image_size is not None and "x" not in args.image_size:
        args.image_size = f"{args.image_size}x{args.image_size}"

    return args
