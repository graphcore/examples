# Dolly 2.0 – The World’s First, Truly Open Instruction-Tuned LLM on IPUs – Inference

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PopXL | NLP | Dolly 2.0 | N/A | Instruction Fine-tuned Text Generation | <p style="text-align: center;">❌ <br>| <p style="text-align: center;">✅ <br> Min. 4 IPU (POD4) required | [Blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) |

[Dolly 2.0](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) is a 12B parameter language model trained and instruction fine-tuned by [Databricks](https://www.databricks.com). By instruction fine-tuning the large language model (LLM), we obtain an LLM better suited for human interactivity. Crucially, Databricks released all code, model weights, and their fine-tuning dataset with an open-source license that permits commercial use. This makes Dolly 2.0 the world's first, truly open-source instruction-tuned LLM. This app requires a minimum of 4 IPUs to run and supports faster inference on POD16.

The best way to run this demo is on Paperspace Gradient's cloud IPUs because everything is already set up for you.

[![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://ipu.dev/t8Jxz1)

## Instructions summary:
Before you can run this model for training or inference you will need to:

1. Install and enable the Poplar SDK (see Poplar SDK setup)
2. Install the Python requirements (see Environment setup)

### Poplar SDK setup
To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
 ```

If no path is provided, then follow these steps:
1. Navigate to your Poplar SDK root directory

2. Enable the Poplar SDK with:
```bash
cd poplar-<OS version>-<SDK version>-<hash>
. enable
```


More detailed instructions on setting up your Poplar environment are available in the [Poplar Quick Start guide] (https://docs.graphcore.ai/projects/poplar-quick-start/en/latest/)

## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to this example's root directory

3. Install the Python requirements with:
```bash
pip3 install -r requirements.txt
```

## Inference

> It is highly recommended to run all inference scripts with the environmental
> variable `POPXL_CACHE_DIR` set to a directory of your choice. This will cache
> parts of the graph compilation, thus speeding up subsequent launches.

To run the benchmarking script, execute the following in the example root:
```shell
python inference.py --config <config_name>
```
where `config_name` is a config found in `config/inference.yml`.

To run an interactive inference script with trained weights, execute the following:
```shell
python run-inference.py --config <config-name>
```

Finally, this example includes a Jupyter notebook. Start a Jupyter server and
navigate to `Dolly2-an-OSS-instruction-LLM.ipynb` to explore Dolly inference in
an interactive environment.

## License
This example is licensed as per the `LICENSE` file at the root of this
repository. This example, when executed, downloads and uses a checkpoint
provided by Databricks which is licensed under the Apache 2.0 License.
