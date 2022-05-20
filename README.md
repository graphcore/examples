# Graphcore code examples

This repository contains sample applications and code examples for use with Graphcore IPUs.

If you are interested in finding out more about Graphcore, including
getting preview access to IPUs to run these examples, please register
your interest [here](https://www.graphcore.ai/product_info)

Please note we are not currently accepting pull requests or issues on this
repository. If you are actively using this repository and want to report any issues, please raise a ticket through the Graphcore support portal: https://www.graphcore.ai/support.

The latest version of the documentation for the Poplar software stack, and other developer resources, is available at https://www.graphcore.ai/developer.

>  The code presented here requires using Poplar SDK 2.5.x

Please install and enable the Poplar SDK following the instructions in the Getting Started guide for your IPU system.

Unless otherwise specified by a LICENSE file in a subdirectory, the LICENSE referenced at the top level applies to the files in this repository.
<br>
<br>

## Repository contents
1. [Computer Vision](#cv)
2. [Natural Language Processing](#nlp)
3. [Speech](#speech)
4. [Graph Neural Network](#gnn)
5. [AI for Simulation](#simulation)
6. [Recommender Systems](#recommender_systems)
7. [Miscellaneous](#miscellaneous)

<br>


### Computer Vision <a name="cv"></a>
| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| ResNet  | Image Classification | Training & Inference | [TensorFlow 1](applications/tensorflow/cnns/) , [TensorFlow 2](applications/tensorflow2/classification/), [PyTorch](applications/pytorch/cnns/)|
| ResNeXt  | Image Classification | Training & Inference | [TensorFlow 1](applications/tensorflow/cnns/) , [PopART (Inference)](applications/popart/resnext_inference)
| EfficientNet | Image Classification | Training & Inference | [TensorFlow 1](applications/tensorflow/cnns/) , [PyTorch](applications/pytorch/cnns/)|
| MobileNet | Image Classification | Inference | [TensorFlow 1](applications/tensorflow/cnns/inference) |
| MobileNetv2 | Image Classification | Inference | [TensorFlow 1](applications/tensorflow/cnns/inference) |
| MobileNetv3 | Image Classification | Training & Inference | [PyTorch](applications/pytorch/cnns/) |
| ViT(Vision Transformer) | Image Classification | Training| [PyTorch](applications/pytorch/vit) |
| DINO | Image Classification | Training| [PyTorch](applications/pytorch/dino) |
| Yolov3 | Object Detection | Training & Inference | [TensorFlow 1](applications/tensorflow/detection/yolov3) |
| Yolov4-P5 | Object Detection | Inference | [PyTorch](applications/pytorch/detection) |
| Faster RCNN | Object Detection | Training & Inference | [PopART](applications/popart/faster-rcnn) |
| EfficientDet | Object Detection | Inference | [TensorFlow 2](applications/tensorflow2/efficientdet) |
| SSD  | Object Detection | Inference | [TensorFlow 1](code_examples/tensorflow/ssd)|
| UNet (Medical) | Image segmentation | Training & Inference | [TensorFlow 2](applications/tensorflow2/unet/)  |
| UNet (Industrial) | Image segmentation | Training | [TensorFlow 1](code_examples/tensorflow/unet_industrial)  |
| miniDALL-E | Generative model in Vision | Training & Inference | [PyTorch](applications/pytorch/miniDALL-E) |
| Neural Image Fields | Neural Radiance Fields | Training | [TensorFlow 2](code_examples/tensorflow2/neural_image_fields)  |
<br>

### Natural Language Processing <a name="nlp"></a>
| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| BERT | NLP | Training & Inference |[TensorFlow 1](applications/tensorflow/bert) , [PyTorch](applications/pytorch/bert) , [PopART](applications/popart/bert), [TensorFlow 2](applications/tensorflow2/bert)|
| Group BERT | NLP | Training |[TensorFlow 1](applications/tensorflow/bert/README.md#GroupBERT_model) |
| Packed BERT | NLP | Training |[PyTorch](applications/pytorch/bert), [PopART](applications/popart/bert) |

<br>


### Speech <a name="speech"></a>
| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| DeepVoice3 | TTS (TextToSpeech) | Training & Inference |[PopART](applications/popart/deep_voice) |
| FastSpeech2 | TTS(TextToSpeech) | Training & Inference | [TensorFlow 2](applications/tensorflow2/fastspeech2/) |
| Conformer | STT(SpeechToText) | Training & Inference | [PopART](applications/popart/conformer_asr), [TensorFlow 1](applications/tensorflow/conformer) , [PyTorch](applications/pytorch/conformer) |
| Transfomer Transducer | STT(SpeechToText) | Training & Inference | [PopART](applications/popart/transformer_transducer) |
<br>

### Graph Neural Network <a name="gnn"></a>
| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| TGN (Temporal Graph Network) | GNN | Training & Inference | [TensorFlow 1](applications/tensorflow/tgn/) |
| MPNN (Message Passing Neural Networks) | GNN | Training & Inference | [TensorFlow 2](code_examples/tensorflow2/message_passing_neural_network) |
| Spektral GNN library with QM9 | GNN | Training | [TensorFlow 2](code_examples/tensorflow2/gnn)  |
| Cluster GCN | GNN | Training & Inference | [TensorFlow 2](applications/tensorflow2/cluster_gcn) |

<br>

### AI for Simulation <a name="simulation"></a>
| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| DeepDriveMD | Biology (Protein folding) | Training | [TensorFlow 2](code_examples/tensorflow2/deep_drive_md)  |
| CosmoFlow example using 3D Convolutions  | Cosmology| Training & Inference | [TensorFlow 1](code_examples/tensorflow/cosmoflow)|

<br>

### Recommender Systems <a name="recommender_systems"></a>
| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| Deep AutoEncoders for Collaborative Filtering | Recommender Systems | Training & Inference | [TensorFlow 1](applications/tensorflow/autoencoder) |
| Click through rate: Deep Interest Network | Recommender Systems | Training & Inference | [TensorFlow 1](applications/tensorflow/click_through_rate) |
| Click through rate: Deep Interest Evolution Network | Recommender Systems | Training & Inference | [TensorFlow 1](applications/tensorflow/click_through_rate) |
<br>

### Miscellaneous <a name="miscellaneous"></a>
| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| RL Policy model | Reinforcement Learning | Training | [TensorFlow 1](applications/tensorflow/reinforcement_learning) |
| MNIST RigL | Dynamic Sparsity | Training | [TensorFlow 1](applications/tensorflow/dynamic_sparsity/mnist_rigl) |
| Autoregressive Language Modelling | Dynamic Sparsity | Training | [TensorFlow 1](applications/tensorflow/dynamic_sparsity/language_modelling) 
| Block-Sparse library  | Sparsity | Training & Inference | [PopART](code_examples/popart/block_sparse) , [TensorFlow 1](code_examples/popart/block_sparse)|
| Sales forecasting | MLP (Multi-Layer Perceptron) | Training | [TensorFlow 1](applications/tensorflow/sales_forecasting) |
| Contrastive Divergence VAE using MCMC methods  | Generative Model | Training | [TensorFlow 1](applications/tensorflow/contrastive_divergence_vae) |
| Monte Carlo Ray Tracing  | Vision | Inference | [Poplar](applications/poplar/monte_carlo_ray_tracing) |
| mcmc  | Statistics | Training & Inference | [TensorFlow 1](code_examples/tensorflow/mcmc)|
| Approximate Bayesian Computation (ABC) COVID-19 | Medical | Inference | [TensorFlow 2](code_examples/tensorflow2/abc_covid_19)  |

<br>
<br>


## Glossary
<br>

### Application examples

The [applications/](applications) folder contains example applications written in different frameworks targeting the IPU. See the READMEs in each folder for details on how to use these applications.
<br>

### Code examples

The [code_examples/](code_examples) folder contains smaller models and code examples. See the READMEs in each folder for details.
<br>

### Tutorials

Tutorials and further code examples can be found in our dedicated [Tutorials repository](https://github.com/graphcore/tutorials).
<br>

### Utilities

The [utils/](utils) folder contains utilities libraries and scripts that are used across the other code examples. This includes:

* [utils/examples_tests](utils/examples_tests) - Common Python helper functions for the repository's unit tests.
* [utils/benchmarks](utils/benchmarks) - Common Python helper functions for running benchmarks on the IPU in different frameworks.

<br>
<br>

## Changelog

### May 2022
- Added those models below to reference models
    - Vision : ViT-pretraining(PyTorch), DINO(PyTorch), EfficientDet-inference(TensorFlow 2), Neural Image Fields (TensorFlow 2)
    - NLP : PackedBERT(PyTorch, PopART), BERT-Large(TensorFlow 2)
    - Speech : FastSpeech2-inference(TensorFlow 2), Conformer-Large(PyTorch)
    - GNN : Cluster GCN(TensorFlow 2)
    - AI for Simulation : DeepDriveMD(TensorFlow 2)

### December 2021
- Added those models below to reference models
    - Vision : miniDALL-E(PyTorch), Faster RCNN(PopART), UNet(TensorFlow 2), ResNet50(TensorFlow 2)
    - NLP : BERT(TensorFlow 2)
    - Speech : FastSpeech2(TensorFlow 2), Transfomer Transducer(PopART), Conformer-Small(PyTorch)
    - GNN : TGN(TensorFlow 1), MPNN(TensorFlow 2)



