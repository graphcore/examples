# Convergence Regression Testing Tool

This tool aims to provide a tractable and automatable mechanism for regression testing the convergence of large models
(e.g. BERT), for which a full training process can take days or weeks.

We do this by capturing a series of checkpoints from a known-good configuration through a full training run. These
serve as the source-of-truth against which to compare future runs. This is referred to as the "gather" stage.

At each checkpoint, we also record key metrics, such as the loss and training accurcy, over a number of subsequent
training steps. This gives us short segments of the overall training curve that can be used to compare a new training
run against the source-of-truth. They are recorded in a JSON manifest file, indexed against the checkpoint for which
they are recorded (with a path relative to the manifest file itself).

In order to test a new variant of the model (or SDK), we then run the model, loading in the checkpoint and running it
for the same number of steps as were recorded. We read-back the same key metrics and compare them to ensure they either
match or improve on the past result (within some margin). This is referred to as the "run" stage.

## Application Requirements

In order to use the convergence harness, an application must allow the user to specify a small selection of arguments.
The names of these arguments are unimportant (this can be specified in the config file), but the functionality must
exist:

For "gather", one must be able to provide arguments for...

- ... the checkpoint output path
- ... the save interval (how many steps/epochs to run between each checkpoint save)
- ... how long to run the model (i.e. a total number of epochs for which to run).

For "run", one must be able to provide arguments for...

- ... the initialisation checkpoint
- ... a starting step/epoch (i.e. fast-forwarding the dataset, logging, etc. to a given point)
- ... how long to run the model (i.e. a total number of epochs for which to run).

Examples of these are shown in `demo_config.yml` and explained further in the ["Configuration" section](#Configuration).

An additional requirement is that the application must log when a checkpoint is saved, and output the metrics to be
measured in subsequent log entries.

The harness uses the "Saving checkpoint" log entry (in whatever form that may be) as a trigger to start recording logs.
It will the extract the required metrics for each log entry thereafter until the required number of steps/epochs have
completed.

## Usage

There are two entry points to the tool, either standalone, using command-line arguments, or from inside your tests
using the `do_convergence_test(...)` helper.

### Standalone

The application takes 2 mandatory arguments:

- `-t`, `--test-config-path`: The path to the test configuration YAML file. See ["Configuration"](#Configuration) for
more details.

- `-s`, `--storage-path`: This is the path used for storing the checkpoints and manifest. In the case of the initial
"gather" stage, this is a writeable output path (directories will be created as required). In the "run" stage, this is
readable path.

There are also two optional arguments:

- `-g`, `--gather`: Sets the application to run in "gather" mode, where the checkpoints and metric manifests are
generated to form a source-of-truth.
- `--manifest-name`: Set the name of the manifest file. This defaults to "convergence_harness_manifest.json". Manifests
are prefixed by the name of the configuration used to gather them (see ["Configuration"](#Configuration)).

Example:

    # Perform the gather step using the demo config
    $ python3 utils/convergence/convergence_harness.py -t utils/convergence/configs/demo_config.yml -s <path_to_convergence_results> -g

    # Now perform the run step
    $ python3 utils/convergence/convergence_harness.py -t utils/convergence/configs/demo_config.yml -s <path_to_convergence_results>

    # Run with a specifically named manifest
    $ python3 utils/convergence/convergence_harness.py -t utils/convergence/configs/demo_config.yml -s <path_to_convergence_results> --manifest-name testing_manifest.json

### Test helper

The `do_convergence_test(...)` method can be called directly from a Python script (such as a Pytest test), and takes
parameters that closely echo the above arguments.

    do_convergence_test(storage_path, manifest_suffix, test_config_path, do_gather=False)

## Configuration

The test configuration is provided as a YAML file. A full example configuration file for the Bert "demo" application is
provided in the file `demo_config.yml`.

The top level index is the test name. This can be used to provide multiple variants of the same application, e.g.

```
resnet18:
    ...
resnet50:
    ...
resnet101:
    ...
```

Within each variant, there are several key headings:

### Directory

`directory`: The directory containing the application, relative to the root of this repository.

### Executable

`executable`: The executable to run as a list of strings. Any command line parameters should be provided in the
[`flags` field](#Flags)

e.g. `$ python3 myapp.py` becomes:

```
executable:
    - "python3"
    - "myapp.py"
```

### Log output stream

`log_output`: Either `stdout` or `stderr` depending on which stream the application logs its output to (typically
`stdout`).

### Log Parsing

This section defines the regex patterns for various variables that one might wish to extract from the logs.

Each variable is provided as a YAML object (where the name of the object is the name of the variable), containing a
single key-value pair defining the regular experession. Special characters should be escaped (i.e. `\s` becomes `\\s`).

This section *must* define the following regex patterns:

- `step_num`: Used to extract the current step/iteration. This is the key for indexing results in the manifest and for
checking if sufficient data has been recorded. This should include a single group containing the step/iteration as an
integer.
- `model_save`: Used to detect when a checkpoint has been saved, and thus when a recording should commence. This
should include a single group containing the path to the checkpoint.

This section should also contain the Regex patterns for one or more metrics which will be recorded and tested against.
These can consist of multiple values, which can be assigned text-labels using the optional `labels` key alongside the
regex - the order of the labels should match the order of the regex groups.

e.g.:

```
log_parsing:
    step_num:
        regex: "Iteration:\\s+(\\d+)"
    model_save:
        regex: "Saving model to: ([^\\s+]+)"
    loss:
        regex: "Loss\\s+\\(MLM\\s+NSP\\):\\s+([\\d\\.]+)\\s+([\\d\\.]+)"
        labels:
            - MLM
            - NSP
```

The above defines the mandatory patterns and one additional one, for the `loss` metric, which consists of two values.
These two values are then given the labels "MLM" and "NSP" and will be stored as such.

_Note: If `num_groups != num_labels` the system falls back to numeric indexing._

### Metrics

This is where the metrics which are used to test the model are defined. Each element of metrics is a YAML object, the
names of which should match the names of the log parsers given above. Each metric takes a margin (a proportion over
the true value) and a comparison operator as a string.

Options for the comparison operators are (all equalities include the aforementioned margin):

- `==`
- `<=`
- `<`
- `>=`
- `>`
- `!=`

For example, following from the log parsing example above, the `loss` metric could be defined as:

```
metrics:
    loss:
        margin: 0.1
        comparison: "<="
```

### Recording Options

These are options specific to the "gather" step.

Currently, only one option is supported (and is mandatory), `steps_to_record_after_save`.

This specifies the absolute number of steps that should be recorded after a checkpoint, regardless of whether a metric
is taken at every step. For example one might record every 5 steps. if `steps_to_record_after_save == 20`, one would
have 4 recorded metric values when the recorder stops.

E.g.:

```
recording:
    steps_to_record_after_save: 3
```

### Flags

This section defines the command line flags for your application. They are divided into 3 broad categories:

- `common`: Flags that should be provided regardless of whether this the harness is in "run" or "gather" mode.
- `gather`: Flags that should only be provided during "gather"
- `run`: Flags that should only be provided during "run"

Flags are given as YAML objects, with 2 fields: `key` and `value`. For key-only flags, provide `null` as the `value`.  
The name of the object is only used by the harness itself to address the flag, whilst `key` and `value` are passed to
the application under test. This slightly verbose structure allows parameter nomenclature to vary between applications.

For most mandatory flags, the actual values passed into the application are decided by the harness, and so a `value`
should not be provided. Instead the `set_in_script` field should be provided and set to `true`. Again, this is to allow
flexible flag nomenclature between applications.

There are a number of mandatory flags in the `gather` and `run` sections, which are required for the harness'
operation. Non-mandatory flags should be provided under the `misc` heading. This is used to delineate those flags
required by the harness from those required by the application.

For the `gather` mode, only one flag is mandatory, and is populated by the harness:

- `checkpoint_output`: The directory into which to write checkpoints.

 For the `run` mode, there are 3 mandatory flags, all of which are again populated by the harness:

- `checkpoint_input`: The checkpoint used to initialise the model for the test.
- `start_step`: The flag that tells the application to fast-forward by a given number of steps/epochs.
- `run_for`: The number of steps/epochs to run for.

The following example shows the mandatory flags, plus some application specific ones:

```
flags:
    common:
        log_steps:
            key: "--steps-per-log"
            value: 1
        config:
            key: "--config"
            value: "configs/demo.json"
    gather:
        checkpoint_output:
            set_in_script: true
            key: "--checkpoint-dir"
        misc:
            save_interval:
                key: "--epochs-per-save"
                value: 5
            epochs_to_run:
                key: "--epochs"
                value: 200
    run:
        checkpoint_input:
            key: "--onnx-checkpoint"
            set_in_script: true
        start_step:
            key: "--continue-training-from-epoch"
            set_in_script: true
        run_for:
            key: "--epochs"
            set_in_script: true
        misc:
            engine_cache:
                key: "--engine-cache"
                value: "__demo_runtest_engine_cache"
```

## The Manifest File

The manifest is stored along with the checkpoints to provide the expected values to the harness during run-mode.

It is a JSON file, containing a single field `ckpt_logs`. This field then contains a dictionary of checkpoint results,
indexed by the step at which each recording was started.

Each result object contains a reference to the checkpoint, the start-step (as a field) and the actual metric
measurements at each step.

E.g. The above config might produce something like the below:

```
{
    "ckpt_logs": {
        "0": {
            "checkpoint": "ckpts/20-04-11-14-23-13/model_0.onnx",
            "start_step": 0,
            "results": {
                "0": {
                    "loss": {
                        "MLM": "9.742",
                        "NSP": "0.718"
                    }
                },
                "1": {
                    "loss": {
                        "MLM": "7.984",
                        "NSP": "0.704"
                    }
                },
                "2": {
                    "loss": {
                        "MLM": "7.336",
                        "NSP": "0.666"
                    }
                }
            }
        },
        "4": {
            "checkpoint": "ckpts/20-04-11-14-23-13/model_4:5.onnx",
            "start_step": 4,
            "results": {
                "5": {
                    "loss": {
                        "MLM": "5.840",
                        "NSP": "0.625"
                    }
                },
                "6": {
                    "loss": {
                        "MLM": "5.477",
                        "NSP": "0.640"
                    }
                },
                "7": {
                    "loss": {
                        "MLM": "5.344",
                        "NSP": "0.650"
                    }
                }
            }
        }
    }
}
 ```

## Running the tests

The tests use pytest. All requirements can be installed using pip and the included `requirements.txt`:

    pip install -r requirements.txt

Then simply run:

    pytest


## File Structure

- `README.md`: This README file.
- `requirements.txt`: Python dependencies file
- `convergence_harness.py`: The main script for the convergence harness
- `convergence_log_recorder.py`: This class implements the functionality to record metrics against
checkpoints and generate the manifest.
- `tests/test_convergence_harness.py`: A set of unit tests for the convergence harness functionality
(~90% coverage).
- `tests/test_utils.py`: Utility functions for the aforementioned unit tests.
- `configs/demo_config.yml`: An example configuration file for the Bert demo application.
