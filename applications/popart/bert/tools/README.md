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

## Schedule Generator
This is a python script to generate learning rate schedules to be used in `.json` config files. For example: 
```
> python tools/schedule_generator.py --function discrete-exp --parameters 0.0006 0.05 --end 130 --interval 10
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

```
See `-h/--help` for more information.