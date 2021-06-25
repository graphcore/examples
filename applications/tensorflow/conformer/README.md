# Conformer: Convolution-augmented Transformer for Speech Recognition

Implementation of the convolutional module from the [Conformer](https://arxiv.org/abs/2005.08100) paper, but with some changes.

1. Replace RNN decoder with [Transformer](https://arxiv.org/pdf/1706.03762v5.pdf) Decoder
2. Add [CTC loss](https://www.cs.toronto.edu/~graves/icml_2006.pdf) for multi task training


## File structure

| File/Folder      | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| data/            | data folder                                                  |
| logs/            | training log folder                                          |
| configs/         | folder for model configs                                     |
| data_loader.py   | dataloader class                                             |
| tests/           | test cases for purest<br />- test_data.py: test case for data<br />- test_model.py: test case for model |
| main.py          | training entrance                                            |
| requirements.txt |                                                              |
| ipu_optimizer.py | adamw optimizer with loss scale                              |
| espnet_docker    | espnet dockerfile for data preparation                       |
|                  |                                                              |


## Quick start guide

1. Download the Poplar SDK

   Download and install the Poplar SDK following the Getting Started guide for your IPU system. Source the enable.sh script for poplar.

2. Configure Python virtual environment

   Create a virtual environment and install the appropriate Graphcore TensorFlow 1.15 wheel from inside the SDK directory:
   ```
   shell
   virtualenv --python python3.6 .bert_venv
   source .bert_venv/bin/activate
   pip3 install -r requirements.txt
   pip3 install <path to the tensorflow-1 wheel from the Poplar SDK>
   ```

3. Prepare the data

   We use ESPNET only for data preparation. You can use the dockerfile, see `espnet_docker/dockerfile`. To build and process the datas (make sure you have `data/` folder):
   ```
   shell
   cd espnet_docker
   docker build --network host -t espnet:data .
   ./run.sh
   ```

   The size of the data is about 60G. And will take about 3 hours(depending on your network).
   for further information see the [ESPNET official docker image](https://espnet.github.io/espnet/docker.html).
   
   The contents of this directory should be:
   ```
   --train_sp/                --> train folder
      |--deltafalse/
         |-- feats.1.ark
         |-- feats.2.arc
         |-- ...
         |-- feats.1.scp
         |-- feats.2.scp
         |-- data.json
   --test/                 --> test folder
      |...
   --dev/                  --> dev folder      
      |...
   --trainsp_units.txt  --> vocab file
   ```

   Set the `data_path` and `dict_path` to the corresponding data paths in the configuration files in the `configs/` directory (egs. `data_path: data/train_sp/deltafalse`, `dict_path: data/trainsp_units.txt`). Ensure that `use_synthetic_data` is false.


## Examples of running the model

Training with sample data.

```
python3 main.py --config configs/train_fp32_kl_loss.yaml
```

## Licensing

The code presented here is licensed under the Apache License Version 2.0, see the LICENSE file in this directory.

This application leverages the Kaldi-ASR framework and ESPNet.

Kaldi-ASR is licensed under Apache 2.0: https://github.com/kaldi-asr/kaldi/blob/master/COPYING

ESPNet is also licensed under Apache 2.0: https://github.com/espnet/espnet/blob/master/LICENSE

