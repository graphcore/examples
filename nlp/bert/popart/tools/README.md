# Bert Tools

## Multi-Checkpoint Validation

This is a wrapper around the main `bert.py` application, which will perform validation over
a series of ONNX checkpoints without requiring recompilation of the graph through Popart/Poplar.

__Note: There is an assumption that all checkpoints come from the same graph.__

The script requires a single argument: `--checkpoint-dir`, which specifies the path to the
directory containing the checkpoints.

The script will attempt to locate the configuration file used in training from the checkpoint
directory, in order to build the correct graph. The user can manually specify its location
using the `--config` argument.

Additional arguments are passed directly through to Bert and can be used to override the
validation config.

### Number of Processes

If desirable, multiple validation threads can be spawned to maximise usage of available IPUs.

This is specified using the optional `--num-processes` command line argument; default is 1.

Since each process will require a separate graph compilation, there is a balance here between
reducing compilation time and validation time. For a small number of checkpoints, a single thread
is probably sufficient.

## Multi-Checkpoint SQuAD

Another wrapper around `bert.py`, this script will run SQuAD fine-tuning against a set of checkpoints
held in the given directory. This only performs the fine-tuning step, no validation. To carry out validation, please see the above Multi-Checkpoint Validation script.

The script takes two arguments:

* `--checkpoint-dir`: The path to the directory containing the pretrained checkpoints to initialise
the fine-tuning run (note that checkpoint discovery is not recursive).

* `--config`: The path to the SQuAD config file

There are two further optional arguments:

* `--model-search-string`: A glob-compatible string that defines the filename pattern to expect for
the pretrained checkpoints.

* `--no-logger-setup`: Convenience function to prevent the logger being set up multiple times if this
script is used as a module rather than run directly.

All other arguments will be forwarded to the Bert argument parser.

Example usage:

    $ python3 tools/bert_multi_checkpoint_squad.py --checkpoint-dir <path_to_checkpoints> --config <path_to_squad_config>

*Note:*  Pretraining checkpoints contain a transposed embedding stored in the ONNX file. Due to the
way checkpoints are loaded to the Popart engine with this script, this will cause the load to fail.
As a solution, we also provide the `transpose_embeddings_only.py` script, which can preprocess the
checkpoint file to tranpose the embedding to match the SQuAD config (see below).

## Transpose Embeddings Script

Pretraining and SQuAD configs differ in their orientation of the embedding tensor. Under normal
execution, the Bert script will transpose the embedding before populating its initialiser; however
when we directly load a checkpoint into the Popart engine, this preprocessing step isn't performed
and Popart will return an error as the tensor shapes don't match.

This script can be used to pre-process the pretrained checkpoints to be compatible with the SQuAD
config. Since it only requires the CPU, it's recommended to parallelise it over as many cores as
possible, e.g.:

    $ python3 tools/transpose_embeddings_only.py --checkpoint-dir <path_to_checkpoints> --output-dir <path_to_store_modified_ckpts> --config <path_to_squad_config> --num-processes 48 --model-search-string "model_*.onnx"

## Multi-Squad Runner Script

To validate fine-tuning over a number of checkpoints requires running the above 3 scripts, so
we provide a helper script in `scripts/run_squad_on_pretraining_ckpts.sh`. This takes two positional
arguments, the path to the directory containing the pretrained checkpoints, and the SQuAD config to
use in fine-tuning.

   $ ./scripts/run_squad_on_pretraining_ckpts.sh <ckpt_dir> <config_path>

## Schedule Generator

This is a python script to generate learning rate schedules to be used in `.json` config files. For example:

    $ python3 tools/schedule_generator.py --function discrete-exp --parameters 0.0006 0.05 --end 130 --interval 10
    {
        "lr_schedule_by_step": {
            "0": 0.0006,
            "10": 0.00057,
            "20": 0.0005414999999999999,
            "30": 0.0005144249999999999,
            "40": 0.0004887037499999999,
            "50": 0.0004642685624999998,
            "60": 0.0004410551343749998,
            "70": 0.0004190023776562498,
            "80": 0.00039805225877343733,
            "90": 0.00037814964583476543,
            "100": 0.00035924216354302717,
            "110": 0.0003412800553658758,
            "120": 0.00032421605259758194,
            "130": 0.00030800524996770287
        }
    }

See `-h/--help` for more information.
