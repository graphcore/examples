# FastPitch
FastPitch for mel-spectrogram generation based on the [fastpitch project](https://github.com/NVIDIA/DeepLearningExamples/tree/793027c9da94eb8449ddda3f33543967269aa480/PyTorch/SpeechSynthesis/FastPitch), optimised for Graphcore's IPU.

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PyTorch | Speech | FastPitch | LJ-Speech | mel-spectrogram generation | <p style="text-align: center;">✅ <br> Min. 16 IPU (POD16) required  | <p style="text-align: center;">❌ | [FastPitch: Parallel Text-to-speech with Pitch Prediction](https://arxiv.org/abs/2006.06873) |

## Instructions summary

1. Install and enable the Poplar SDK (see Poplar SDK setup)

2. Install the system and Python requirements (see Environment setup)

3. Download the LJ-Speech dataset (See Dataset setup)


## Poplar SDK setup
To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
```

If no path is provided, then follow these steps:
1. Navigate to your Poplar SDK root directory

2. Enable the Poplar SDK with:
```bash
cd poplar-<OS version>-<SDK version>-<hash>
. enable.sh
```

3. Additionally, enable PopART with:
```bash
cd popart-<OS version>-<SDK version>-<hash>
. enable.sh
```

More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to this examples root directory

3. Install the Python requirements
```bash
pip3 install -r requirements.txt
```

More detailed instructions on setting up your PyTorch environment are available in the [PyTorch quick start guide](https://docs.graphcore.ai/projects/pytorch-quick-start).

## Dataset setup
Download the LJ-Speech dataset from [the source](https://keithito.com/LJ-Speech-Dataset/) or using the scripts provided:
```bash
bash scripts/download_dataset.sh
```

And preprocess the dataset using:
```bash
bash scripts/download_tacotron2.sh
bash scripts/prepare_dataset.sh
```
This may take a long time (about 10 hours) because it uses tacotron2 on CPU to do duration prediction.

Disk space required: 2.6GB

```bash
.
├──
 directories, file
```


## Running and benchmarking
To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. The benchmarks are provided in the `benchmarks.yml` file in this example's root directory.

For example:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).


## License

The example in this directory is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file in this directory.
This license applies to the following files:
- dataset.py
- ipu_config.py
- pipeline_base_model.py
- pics/loss_curve.jpeg
- platform/FastPitch_FP32_2IPU.sh
- README.md
- tests/test_layers.py
- tests/unit_tester.py

The other files come from the original [FastPitch](https://github.com/NVIDIA/DeepLearningExamples/tree/793027c9da94eb8449ddda3f33543967269aa480/PyTorch/SpeechSynthesis/FastPitch) repository which is licensed under the 3-clause BSD license. The license from that repository is reproduced here as [ORIGINAL-FASTPITCH-LICENSE](ORIGINAL-FASTPITCH-LICENSE).
Some of the files in the `common/text` subdirectory are based on content from https://github.com/keithito/tacotron which is licensed under the MIT license. See the [LICENSE](common/text/LICENSE) file in that subdirectory for details.
