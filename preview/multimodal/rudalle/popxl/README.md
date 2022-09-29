PopXL ruDALL-E XL inference example using tensor model parallelism
---

Inference implementation of ruDALL-E XL (1.3B parameters) in PopXL for IPU. It uses tensor model parallelism on 8 IPUs. This example is based on the model provided by the [`ruDALL-E`](https://github.com/ai-forever/ru-dalle) library. The ruDALL-E model is based on the original paper [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092).

## Environment Setup

First, Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` scripts for poplar and popART, see more details in [SDK installation](https://docs.graphcore.ai/projects/ipu-pod-getting-started/en/latest/installation.html#sdk-installation).

Then, create a virtual environment and install the required packages.

```console
virtualenv venv -p python3.6
source venv/bin/activate

pip install -r requirements.txt
```

## File structure <a name="file_structure"></a>

| File | Description |
|--------------------------- | --------------------------------------------------------------------- |
| `config/` | Contains configuration options for inference. |
| `modelling/` | Implements networks in ruDALL-E model. `modeling_cached_TP.py` and `image_decoder.py` present the implementations of transformer and image decoder. |
| `tests/` | Unit tests. |
| `utils/` | Helper functions |
| `inference.py`| Contains the execution scheme for inference. If executed directly, use generated data to do inference benchmarking. |
| `run_inference.py` | Run ruDALL-E XL inference on IPU using HF weights `sberbank-ai/rudalle-Malevich`. |

## Run the application

Setup your environment as explained above, then you can run ruDALL-E inference:

```console
python run_inference.py --text 'The sunrise'
```

The generated image has same file name with input text, i.e., "The sunrise.png". The generated image of each run is different because the seed are random.
