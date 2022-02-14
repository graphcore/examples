# Graphcore code examples

This repository contains sample applications and code examples for use with Graphcore IPUs.

If you are interested in finding out more about Graphcore, including
getting preview access to IPUs to run these examples, please register
your interest [here](https://www.graphcore.ai/product_info).

Please note we are not currently accepting pull requests or issues on this
repository. If you are actively using this repository and want to report any issues, please raise a ticket through the Graphcore support portal: https://www.graphcore.ai/support.

The latest version of the documentation for the Poplar software stack, and other developer resources, is available at https://www.graphcore.ai/developer.

>  The code presented here requires using Poplar SDK 2.4.x

Please install and enable the Poplar SDK following the instructions in the Getting Started guide for your IPU system.

Unless otherwise specified by a LICENSE file in a subdirectory, the LICENSE referenced at the top level applies to the files in this repository.

## Repository contents

### Application examples

The [applications/](applications) folder contains example applications written in different frameworks targeting the IPU. See the READMEs in each folder for details on how to use these applications.

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| ResNet  | Image Classifcation | Training & Inference | [TensorFlow 1](applications/tensorflow/cnns/) , [TensorFlow 2](applications/tensorflow2/classification/), [PyTorch](applications/pytorch/cnns/)|
| ResNeXt  | Image Classifcation | Training & Inference | [TensorFlow 1](applications/tensorflow/cnns/) , [PopART (Inference)](applications/popart/resnext_inference)
| EfficientNet | Image Classifcation | Training & Inference | [TensorFlow 1](applications/tensorflow/cnns/) , [PyTorch](applications/pytorch/cnns/)|
| MobileNet | Image Classifcation | Inference | [TensorFlow 1](applications/tensorflow/cnns/inference) |
| MobileNetv2 | Image Classifcation | Inference | [TensorFlow 1](applications/tensorflow/cnns/inference) |
| MobileNetv3 | Image Classifcation | Training & Inference | [PyTorch](applications/pytorch/cnns/) |
| ViT(Vision Transformer) | Image Classifcation | Training| [PyTorch](applications/pytorch/vit) |
| Yolov3 | Object Detection | Training & Inference | [TensorFlow 1](applications/tensorflow/detection/yolov3) |
| Yolov4-P5 | Object Detection | Inference | [PyTorch](applications/pytorch/detection) |
| Faster RCNN | Object Detection | Training & Inference | [PopART](applications/popart/faster-rcnn) |
| UNet (Medical) | Image segmentation | Training & Inference | [TensorFlow 2](applications/tensorflow2/unet/)  |
| miniDALL-E | Generative model in Vision | Training & Inference | [PyTorch](applications/pytorch/miniDALL-E) |
| BERT | NLP | Training & Inference |[TensorFlow 1](applications/tensorflow/bert) , [PyTorch](applications/pytorch/bert) , [PopART](applications/popart/bert), [TensorFlow 2](applications/tensorflow2/bert)|
| DeepVoice3 | TTS (TextToSpeech) | Training & Inference |[PopART](applications/popart/deep_voice) |
| FastSpeech2 | TTS(TextToSpeech) | Training & Inference | [TensorFlow 2](applications/tensorflow2/fastspeech2/) |
| Conformer | STT(SpeechToText) | Training & Inference | [PopART](applications/popart/conformer_asr) |
| Conformer with Transformer | STT(SpeechToText) | Training & Inference | [TensorFlow 1](applications/tensorflow/conformer) , [PyTorch](applications/pytorch/conformer) |
| Transfomer Transducer | STT(SpeechToText) | Training & Inference | [PopART](applications/popart/transformer_transducer) |
| TGN (Temporal Graph Network) | GNN | Training & Inference | [TensorFlow 1](applications/tensorflow/tgn/) |
| MPNN (Message Passing Neural Networks) | GNN | Training & Inference | [TensorFlow 2](code_examples/tensorflow2/message_passing_neural_network) |
| Deep AutoEncoders for Collaborative Filtering | Recommender Systems | Training & Inference | [TensorFlow 1](applications/tensorflow/autoencoder) |
| Click through rate: Deep Interest Network | Recommender Systems | Training & Inference | [TensorFlow 1](applications/tensorflow/click_through_rate) |
| Click through rate: Deep Interest Evolution Network | Recommender Systems | Training & Inference | [TensorFlow 1](applications/tensorflow/click_through_rate) |
| RL Policy model | Reinforcement Learning | Training | [TensorFlow 1](applications/tensorflow/reinforcement_learning) |
| MNIST RigL | Dynamic Sparsity | Training | [TensorFlow 1](applications/tensorflow/dynamic_sparsity/mnist_rigl) |
| Autoregressive Language Modelling | Dynamic Sparsity | Training | [TensorFlow 1](applications/tensorflow/dynamic_sparsity/language_modelling) |
| Sales forecasting | MLP (Multi-Layer Perceptron) | Training | [TensorFlow 1](applications/tensorflow/sales_forcasting/language_modelling) |
| Contrastive Divergence VAE using MCMC methods  | Generative Model | Training | [TensorFlow 1](applications/tensorflow/contrastive_divergence_vae) |
| Monte Carlo Ray Tracing  | Vision | Inference | [Poplar](applications/poplar/monte_carlo_ray_tracing) |



### Code examples

The [code_examples/](code_examples) folder contains smaller models and code examples. See the READMEs in each folder for details.

### Tutorials

Tutorials and further code examples can be found in our dedicated [Tutorials repository](https://github.com/graphcore/tutorials).

### Utilities

The [utils/](utils) folder contains utilities libraries and scripts that are used across the other code examples. This includes:

* [utils/examples_tests](utils/examples_tests) - Common Python helper functions for the repository's unit tests.
* [utils/benchmarks](utils/benchmarks) - Common Python helper functions for running benchmarks on the IPU in different frameworks.


## Changelog

December 2021:
- Added those models below to reference models
    - Vision : miniDALL-E(PyTorch), Faster RCNN(PopART), UNet(TensorFlow 2), ResNet50(TensorFlow 2)
    - NLP : BERT(TensorFlow 2)
    - TTS/STT : FastSpeech2(TensorFlow 2), Transfomer Transducer(PopART), Conformer with Transformer(PyTorch)
    - GNN : TGN(TensorFlow1), MPNN(TensorFlow 2)



