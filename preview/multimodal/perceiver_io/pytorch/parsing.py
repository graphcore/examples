# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import sys
import dataclasses
import json
from pathlib import Path
from typing import Any, NewType, Tuple, List

from transformers import HfArgumentParser

from configs.hparams import DatasetArguments, ModelArguments, PerceiverTrainingArguments


def _yaml_to_string_list(config: dict) -> List[str]:
    s_list = []
    for arg, value in config.items():
        s_list += [f"--{arg}"]
        if type(value) == list:
            s_list += [str(element) for element in value]
        else:
            s_list += [str(value)]
    return s_list


def parse_arguments():
    DataClass = NewType("DataClass", Any)

    class ExtendedArgumentParser(HfArgumentParser):
        def parse_args_into_dataclasses(self) -> Tuple[DataClass, ...]:
            # parse args from a terminal
            terminal_args = self.parse_args(args=sys.argv[1:] + ["--output_dir", "/tmp/perceiver-io/"])
            # load args from a file
            if terminal_args.config is not None:
                fargs = json.loads(Path(terminal_args.config).read_text())
            # in case of duplicate arguments the first one has precedence
            # terminal args have precedence over file args
            args = _yaml_to_string_list(fargs) + sys.argv[1:]

            namespace, remaining_args = self.parse_known_args(args=args)
            outputs = []
            for dtype in self.dataclass_types:
                keys = {f.name for f in dataclasses.fields(dtype) if f.init}
                inputs = {k: v for k, v in vars(namespace).items() if k in keys}
                for k in keys:
                    delattr(namespace, k)
                obj = dtype(**inputs)
                outputs.append(obj)

            if len(namespace.__dict__) > 0:
                # additional namespace.
                outputs.append(namespace)
            else:
                if remaining_args:
                    raise ValueError(
                        f"Some specified arguments are not used " f"by the HfArgumentParser: {remaining_args}"
                    )
                return (*outputs,)

    parser = ExtendedArgumentParser((ModelArguments, DatasetArguments, PerceiverTrainingArguments))
    return parser.parse_args_into_dataclasses()
