# Graphcore Application examples

This repository contains a catalogue of application examples that have been optimised to run on Graphcore IPUs for both training and inference. Access reproducible code for a wide range of popular models covering NLP, Computer Vision, Speech, Multimodal, GNNs, AI for Simulation, Recommender Systems, and more. This includes a selection of models that achieve state of the art performance on IPUs, as well as code examples for self-learning.

Run models out-the-box on IPUs integrated with popular ML frameworks and libraries:

![Snip 2022-07-05 20 04 06](https://user-images.githubusercontent.com/81682248/177397772-4b671628-a7f4-4d8f-849d-2c5b54dba1de.png)

To see what's new and easily filter applications by domain and framework, please check out our [Model Garden](https://www.graphcore.ai/resources/model-garden) :tulip:.

For more detailed benchmark information, please visit our [Performance Results page](https://www.graphcore.ai/performance-results).

> The code presented here requires using Poplar SDK 3.2.x, and has been tested using Ubuntu 20.04 and Python 3.8

Please install and enable the Poplar SDK following the instructions in the [Getting Started](https://docs.graphcore.ai/en/latest/getting-started.html#pod-system-getting-started-guides) guide for your IPU system.

## Developer Resources
- [Documentation](https://docs.graphcore.ai/en/latest/): Explore our software documentation, user guides, and technical notes
- [Tutorials](https://github.com/graphcore/tutorials/tree/master/tutorials): Hands-on code tutorials, simple application and feature examples
- [How-to Videos](https://www.graphcore.ai/resources/how-to-videos): Watch practical how-to videos and demos by Graphcore engineers
- [Research Papers](https://www.graphcore.ai/resources/research-papers): Read publications from Graphcore's Research team and IPU innovators

## Support

If you encounter a problem or want to suggest an improvement to our example application please raise a Github issue, contact us at
[support@graphcore.ai](mailto:support@graphcore.ai?subject=Applications%20Feedback), or get in touch through the #help channel of our slack community!

[![Join our Slack Community](https://img.shields.io/badge/Slack-Join%20Graphcore's%20Community-blue?style=flat-square&logo=slack)](https://www.graphcore.ai/join-community)

If you require POD128/256 setup and configuration for our applications, please contact [our engineering support](https://www.graphcore.ai/support).
<br>


## Repository contents
1. [Computer Vision](#cv)
2. [Natural Language Processing](#nlp)
3. [Speech](#speech)
4. [Multimodal](#multimodal)
5. [Graph Neural Network](#gnn)
6. [AI for Simulation](#simulation)
7. [Recommender Systems](#recommender_systems)
8. [Reinforcement Learning](#rl)
9. [Sparsity](#sparsity)
10. [Probability](#probability)
11. [Miscellaneous](#miscellaneous)
12. [Archived](#archived)

<br>


### <img width="30" src="https://user-images.githubusercontent.com/81682248/177352641-89d12db1-45df-4403-8308-c6b9015a027d.png"></a> Computer Vision <a name="cv"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| ResNet  | Image Classification | Training & Inference | [TensorFlow 2](vision/cnns/tensorflow2/), [PyTorch](vision/cnns/pytorch/), [PyTorch Lightning](https://github.com/graphcore/pytorch-lightning-examples/tree/release/applications)|
| EfficientNet | Image Classification | Training & Inference | [PyTorch](vision/cnns/pytorch/), [PyTorch Lightning](https://github.com/graphcore/pytorch-lightning-examples/tree/release/applications)|
| MobileNetv3 | Image Classification | Training & Inference | [PyTorch](vision/cnns/pytorch/) |
| ViT(Vision Transformer) | Image Classification | Training | [PyTorch](vision/vit/pytorch/), [Hugging Face Optimum](https://huggingface.co/Graphcore/vit-base-ipu) |
| DINO | Image Classification | Training | [PyTorch](vision/dino/pytorch) |
| Swin | Image Classification | Training | [PyTorch](vision/swin/pytorch)  |
| MAE (Masked AutoEncoder) | Image Classification | Training | [PyTorch](vision/mae/pytorch)  |
| Yolov4-P5 | Object Detection | Inference | [PyTorch](vision/yolo_v4/pytorch) |
| EfficientDet | Object Detection | Inference | [TensorFlow 2](vision/efficientdet/tensorflow2) |
| UNet (Medical) | Image segmentation | Training & Inference | [TensorFlow 2](vision/unet_medical/tensorflow2)  |
| Neural Image Fields | Neural Radiance Fields | Training | [TensorFlow 2](vision/neural_image_fields/tensorflow2)  |
<br>

### <img width="30" src="https://user-images.githubusercontent.com/81682248/177355208-a49a2bba-dd4d-4467-b135-9023279e8f01.png"></a> Natural Language Processing <a name="nlp"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| BERT | NLP | Training & Inference | [PyTorch](nlp/bert/pytorch) , [TensorFlow 2](nlp/bert/tensorflow2/), [PopXL](nlp/bert/popxl), [PaddlePaddle](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/bert/static_ipu), [Hugging Face Optimum](https://huggingface.co/Graphcore/bert-large-ipu)|
| Packed BERT | NLP | Training |[PyTorch](nlp/bert/pytorch) |
| GPT2 | NLP | Training |[PyTorch](nlp/gpt2/pytorch) , [Hugging Face Optimum](https://huggingface.co/Graphcore/gpt2-medium-ipu) |
| GPTJ | NLP | Training |[PopXL](nlp/gpt_j/popxl)|
| GPT3-2.7B | NLP | Training |[PopXL](nlp/gpt3_2.7B/popxl) |
| GPT3-175B | NLP | Training |[PopXL](nlp/gpt3_175B/popxl) |
| RoBERTa | NLP | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/roberta-large-ipu)|
| DeBERTa | NLP | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/deberta-base-ipu)|
| HuBERT | NLP | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/hubert-base-ipu)|
| BART | NLP | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/bart-base-ipu)|
| T5 | NLP | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/t5-small-ipu)|
| Bloom | NLP | Inference |[PopXL](nlp/bloom/popxl) |

<br>


### <img width="30" src="https://user-images.githubusercontent.com/81682248/177355502-87b09860-d323-438a-a0a2-247b8f6e9349.png"></a> Speech <a name="speech"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| Fastpitch | TTS (TextToSpeech) | Training | [PyTorch](speech/fastpitch/pytorch) |
| Conformer | STT(SpeechToText) | Training & Inference | [PyTorch](speech/conformer/pytorch)|
| Wav2Vec2 | STT(SpeechToText) | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/wav2vec2-base-ipu)|

<br>

### <img width="30" src="https://user-images.githubusercontent.com/81682248/177357173-57e4cc6f-cff3-43a9-bd40-dcf3616f1fa1.png"></a> Multimodal <a name="multimodal"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| miniDALL-E | multimodal | Training | [PyTorch](multimodal/mini_dalle/pytorch) |
| CLIP | multimodal | Training |[PyTorch](multimodal/CLIP/pytorch)|
| LXMERT | multimodal | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/lxmert-base-ipu)|
| Frozen in time | multimodal | Training & Inference |[PyTorch](multimodal/frozen_in_time/pytorch)|

<br>

### <img width="25" src="https://user-images.githubusercontent.com/81682248/177357459-84ed7863-6477-4d8f-b63e-3db6c2ad405c.png"></a> Graph Neural Network <a name="gnn"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| MPNN (Message Passing Neural Networks) | GNN | Training & Inference | [PyTorch Geometric](gnn/message_passing/pytorch_geometric) , [TensorFlow 2](gnn/message_passing/tensorflow2) |
| Spektral GNN library with QM9 | GNN | Training | [TensorFlow 2](gnn/spektral/tensorflow2)  |
| Cluster GCN | GNN | Training & Inference | [PyTorch Geometric](gnn/cluster_gcn/pytorch_geometric) , [TensorFlow 2](gnn/cluster_gcn/tensorflow2) |
| TGN (Temporal Graph Networks) | GNN | Training | [PyTorch](gnn/tgn/pytorch) |
| NBFNet | GNN | Training & Inference | [PyTorch Geometric](gnn/nbfnet/pytorch_geometric) |
| SchNet | GNN | Training & Inference | [PyTorch Geometric](gnn/schnet/pytorch_geometric) |
| GPS++ - OGB-LSC PCQM4Mv2 competition submission | GNN | Training & Inference | [TensorFlow 2](gnn/ogb_lsc_pcqm4mv2/tensorflow2) |

<br>

### <img width="25" src="https://user-images.githubusercontent.com/81682248/177359725-f8b1c268-ddbb-41c5-a037-16168564cacc.png"></a> AI for Simulation <a name="simulation"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| Approximate Bayesian Computation (ABC) COVID-19 | Medical | Inference | [TensorFlow 2](ai_for_simulation/abc_covid_19/tensorflow2)  |


<br>


## Benchmarking tools
To easily run the examples with tested and optimised configurations and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), you can use the examples-utils benchmarking module, which comes with every example when you install its requirements. To use this simple, shared interface for almost any of the examples provided here, locate and look through the example's `benchmarks.yml` file and run:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).

<br>

## PopVision® Tools
Visualise your code's inner workings with a user-friendly, graphical interface to optimise your machine learning models.

[Download](https://www.graphcore.ai/developer/popvision-tools) PopVision to analyse IPU performance and utilisation.

<br>

## Utilities
The [utils/](utils) folder contains utilities libraries and scripts that are used across the other code examples. This includes:

* [utils/examples_tests](utils/examples_tests) - Common Python helper functions for the repository's unit tests
* [utils/benchmarks](utils/benchmarks) - Common Python helper functions for running benchmarks on the IPU in different frameworks

<br>

## License
Unless otherwise specified by a LICENSE file in a subdirectory, the LICENSE referenced at the top level applies to the files in this repository.

<br>

## Changelog

<details>
<summary>March 2023</summary>
<br>

*  Added this model below to reference models
    *  GNN: NBFNet (PyTorch Geometric), SchNet (PyTorch Geometric), Cluster-GCN (PyTorch Geometric), GIN (PyTorch Geometric), GPS++ - OGB-LSC PCQM4Mv2 competition submission (TensorFlow 2)
    *  NLP : GPT3_175B (PopXL), Bloom (PopXL)
*  Removed all PopART applications, as well as the following:
    * Miscellaneous: Monte-Carlo ray tracing
    * AI for simulation: DeepDriveMD
    * (Preview) Multimodel: ruDalle
    * Speech: FastSpeech2
    * Vision: ResNeXt inference
* Moved the contents of the [Graphcore/tutorials](https://github.com/graphcore/tutorials) repository into this repository (PopART tutorials have also been removed)
</details>

<details>
<summary>Dec 2022</summary>
<br>

*  Added this model below to reference models
    *  GNN: TGN (PyTorch)
*  Deprecating all PopART applications. Support will be removed in the next release.
*  Removed all TensorFlow 1 applications.
*  Ubuntu 18.04 no longer supported.
</details>

<details>
<summary>Sep 2022</summary>
<br>

*  Added those models below to reference models
    *  Vision : MAE (PyTorch), G16 EfficientNet (PyTorch)
    *  NLP : GPTJ (PopXL), GPT3-2.7B (PopXL)
    *  Multimodal : Frozen in time (PyTorchs), ruDalle(Preview) (PopXL)
*  Deprecating all TensorFlow 1 applications. Support will be removed in the next release.
</details>

<details>
<summary>Aug 2022</summary>
<br>

* Change the folder name of models
    * NLP : from gpt to gpt2
    * Speech : from wenet-conformer to conformer

</details>
<details>
<summary>July 2022</summary>
<br>

*  Major reorganisation of all the apps so that they are arranged as: problem domain / model / framework.
*  Problem domains: Vision, NLP, Speech, GNN, Sparsity, AI for Simultation, Recomender systems, Reinforcement learning, Probability, Multimodal, and Miscellaneous.
*  Added those models below to reference models
    *  Vision : Swin (PyTorch) , ViT (Hugging Face Optimum)
    *  NLP : GPT2 Small/Medium/Large (PyTorch), BERT-Base/Large (PopXL), BERT-Base(PaddlePaddle), BERT-Base/Large(Hugging Face Optimum), GPT2 Small/Medium (Hugging Face Optimum), RoBERTa Base/Large(Hugging Face Optimum), DeBERTa(Hugging Face Optimum), HuBERT(Hugging Face Optimum), BART(Hugging Face Optimum), T5 small(Hugging Face Optimum)
    *  Speech : Fastpitch (PyTorch), WeNet-Conformer-Medium(PyTorch) ,Wav2Vec2(Hugging Face Optimum)
    *  Multimodal : CLIP (PyTorch), LXMERT(Hugging Face Optimum)
    *  AI for Simulation : et0(TensorFlow 1)
* Removed Conformer-small/large (PyTorch)
* Archived Minigo (TensorFlow 1)

</details>


<details>
<summary>May 2022</summary>
<br>

*  Added those models below to reference models
    *  Vision : ViT-pretraining(PyTorch), DINO(PyTorch), EfficientDet-inference(TensorFlow 2), Neural Image Fields (TensorFlow 2)
    *  NLP : PackedBERT(PyTorch, PopART), BERT-Large(TensorFlow 2)
    *  Speech : FastSpeech2-inference(TensorFlow 2), Conformer-Large(PyTorch)
    *  GNN : Cluster GCN(TensorFlow 2)
    *  AI for Simulation : DeepDriveMD(TensorFlow 2)

</details>

<details>
<summary>December 2021</summary>
<br>

*  Added those models below to reference models
    *  Vision : miniDALL-E(PyTorch), Faster RCNN(PopART), UNet(TensorFlow 2), ResNet50(TensorFlow 2)
    *  NLP : BERT(TensorFlow 2)
    *  Speech : FastSpeech2(TensorFlow 2), Transformer Transducer(PopART), Conformer-Small(PyTorch)
    *  GNN : TGN(TensorFlow 1), MPNN(TensorFlow 2)
</details>

<br>

## Connect with us
<p align="center">
  <a href="https://www.graphcore.ai/join-community"><img src="https://img.shields.io/badge/Slack-4A154B.svg?style=for-the-badge&logo=Slack&logoColor=white"/></a>
  <a href="https://twitter.com/graphcoreai"><img src="https://img.shields.io/badge/Twitter-1DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white"/></a>
  <a href="https://www.linkedin.com/company/graphcore"><img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white"/></a>
  <a href="http://www.facebook.com/pages/Graphcore/890447934394683"><img src="https://img.shields.io/badge/Facebook-1877F2.svg?style=for-the-badge&logo=Facebook&logoColor=white"/></a>
  <a href="https://www.youtube.com/c/Graphcore"><img src="https://img.shields.io/badge/YouTube-FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white"/></a>
  <a href="https://medium.com/graphcore"><img src="https://img.shields.io/badge/Medium-000000.svg?style=for-the-badge&logo=Medium&logoColor=white"/></a>
</p>