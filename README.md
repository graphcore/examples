# Graphcore Application examples

This repository contains a catalogue of application examples that have been optimised to run on Graphcore IPUs for both training and inference. Access reproducible code for a wide range of popular models covering NLP, Computer Vision, Speech, Multimodal, GNNs, AI for Simulation, Recommender Systems, and more. This includes a selection of models that achieve state of the art performance on IPUs, as well as code examples for self-learning.

Run models out-the-box on IPUs integrated with popular ML frameworks and libraries:

![Snip 2022-07-05 20 04 06](https://user-images.githubusercontent.com/81682248/177397772-4b671628-a7f4-4d8f-849d-2c5b54dba1de.png)

To see what's new and easily filter applications by domain and framework, please check out our [Model Garden](https://www.graphcore.ai/resources/model-garden) :tulip:.

For more detailed benchmark information, please visit our [Performance Results page](https://www.graphcore.ai/performance-results).

> The code presented here requires using Poplar SDK 2.6.x

Please install and enable the Poplar SDK following the instructions in the [Getting Started](https://docs.graphcore.ai/en/latest/getting-started.html#pod-system-getting-started-guides) guide for your IPU system.

Please contact [our engineering support](https://www.graphcore.ai/support) if you want to have POD128/256 setup and configuration for our applications
<br>
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
| ResNet  | Image Classification | Training & Inference | [TensorFlow 1](vision/cnns/tensorflow1/) , [TensorFlow 2](vision/cnns/tensorflow2/), [PyTorch](vision/cnns/pytorch/), [PyTorch Lightning](https://github.com/graphcore/pytorch-lightning-examples/tree/release/applications)|
| ResNeXt  | Image Classification | Training & Inference | [TensorFlow 1](vision/cnns/tensorflow1/) , [PopART (Inference)](vision/resnext_inference/popart)
| EfficientNet | Image Classification | Training & Inference | [TensorFlow 1](vision/cnns/tensorflow1/) , [PyTorch](vision/cnns/pytorch/), [PyTorch Lightning](https://github.com/graphcore/pytorch-lightning-examples/tree/release/applications)|
| MobileNet | Image Classification | Inference | [TensorFlow 1](vision/cnns/tensorflow1/inference) |
| MobileNetv2 | Image Classification | Inference | [TensorFlow 1](vision/cnns/tensorflow1/inference) |
| MobileNetv3 | Image Classification | Training & Inference | [PyTorch](vision/cnns/pytorch/) |
| ViT(Vision Transformer) | Image Classification | Training| [PyTorch](vision/vit/pytorch/), [Hugging Face Optimum](https://huggingface.co/Graphcore/vit-base-ipu) |
| DINO | Image Classification | Training| [PyTorch](vision/dino/pytorch) |
| Swin | Image Classification | Training | [PyTorch](vision/swin/pytorch)  |
| Yolov3 | Object Detection | Training & Inference | [TensorFlow 1](vision/yolo_v3/tensorflow1) |
| Yolov4-P5 | Object Detection | Inference | [PyTorch](vision/yolo_v4/pytorch) |
| Faster RCNN | Object Detection | Training & Inference | [PopART](vision/faster_rcnn/popart) |
| EfficientDet | Object Detection | Inference | [TensorFlow 2](vision/efficientdet/tensorflow2) |
| SSD  | Object Detection | Inference | [TensorFlow 1](vision/ssd/tensorflow1)|
| UNet (Medical) | Image segmentation | Training & Inference | [TensorFlow 2](vision/unet_medical/tensorflow2)  |
| UNet (Industrial) | Image segmentation | Training | [TensorFlow 1](vision/unet_industrial/tensorflow1)  |
| Neural Image Fields | Neural Radiance Fields | Training | [TensorFlow 2](vision/neural_image_fields/tensorflow2)  |
<br>

### <img width="30" src="https://user-images.githubusercontent.com/81682248/177355208-a49a2bba-dd4d-4467-b135-9023279e8f01.png"></a> Natural Language Processing <a name="nlp"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| BERT | NLP | Training & Inference |[TensorFlow 1](nlp/bert/tensorflow1) , [PyTorch](nlp/bert/pytorch) , [PopART](nlp/bert/popart), [TensorFlow 2](nlp/bert/tensorflow2/), [PopXL](nlp/bert/popxl), [PaddlePaddle](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/bert/static_ipu), [Hugging Face Optimum](https://huggingface.co/Graphcore/bert-large-ipu)|
| Group BERT | NLP | Training |[TensorFlow 1](nlp/bert/tensorflow1/README.md#GroupBERT_model) |
| Packed BERT | NLP | Training |[PyTorch](nlp/bert/pytorch), [PopART](nlp/bert/popart) |
| GPT2 | NLP | Training |[PyTorch](nlp/gpt/pytorch) , [Hugging Face Optimum](https://huggingface.co/Graphcore/gpt2-medium-ipu) |
| RoBERTa | NLP | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/roberta-large-ipu)|
| DeBERTa | NLP | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/deberta-base-ipu)|
| HuBERT | NLP | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/hubert-base-ipu)|
| BART | NLP | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/bart-base-ipu)|
| T5 | NLP | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/t5-small-ipu)|


<br>


### <img width="30" src="https://user-images.githubusercontent.com/81682248/177355502-87b09860-d323-438a-a0a2-247b8f6e9349.png"></a> Speech <a name="speech"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| DeepVoice3 | TTS (TextToSpeech) | Training & Inference |[PopART](speech/deepvoice3/popart) |
| FastSpeech2 | TTS(TextToSpeech) | Training & Inference | [TensorFlow 2](speech/fastspeech2/tensorflow2) |
| Fastpitch | TTS (TextToSpeech) | Training | [PyTorch](speech/fastpitch/pytorch) |
| Conformer | STT(SpeechToText) | Training & Inference | [PopART](speech/conformer/popart), [TensorFlow 1](speech/conformer/tensorflow1) |
| Transfomer Transducer | STT(SpeechToText) | Training & Inference | [PopART](speech/transformer_transducer/popart) |
| WeNet-Conformer | STT(SpeechToText) | Training | [PyTorch](speech/wenet_conformer/pytorch) |
| Wav2Vec2 | STT(SpeechToText) | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/wav2vec2-base-ipu)|

<br>

### <img width="30" src="https://user-images.githubusercontent.com/81682248/177357173-57e4cc6f-cff3-43a9-bd40-dcf3616f1fa1.png"></a> Multimodal <a name="multimodal"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| miniDALL-E | multimodal | Training | [PyTorch](multimodal/mini_dalle/pytorch) |
| CLIP | multimodal | Training |[PyTorch](multimodal/CLIP/pytorch)|
| LXMERT | multimodal | Training | [Hugging Face Optimum](https://huggingface.co/Graphcore/lxmert-base-ipu)|


<br>

### <img width="25" src="https://user-images.githubusercontent.com/81682248/177357459-84ed7863-6477-4d8f-b63e-3db6c2ad405c.png"></a> Graph Neural Network <a name="gnn"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| TGN (Temporal Graph Network) | GNN | Training & Inference | [TensorFlow 1](gnn/tgn/tensorflow1) |
| MPNN (Message Passing Neural Networks) | GNN | Training & Inference | [TensorFlow 2](gnn/message_passing/tensorflow2) |
| Spektral GNN library with QM9 | GNN | Training | [TensorFlow 2](gnn/spektral/tensorflow2)  |
| Cluster GCN | GNN | Training & Inference | [TensorFlow 2](gnn/cluster_gcn/tensorflow2) |

<br>

### <img width="25" src="https://user-images.githubusercontent.com/81682248/177359725-f8b1c268-ddbb-41c5-a037-16168564cacc.png"></a> AI for Simulation <a name="simulation"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| DeepDriveMD | Biology (Protein folding) | Training | [TensorFlow 2](ai_for_simulation/deep_drive_md/tensorflow2)  |
| CosmoFlow example using 3D Convolutions  | Cosmology| Training & Inference | [TensorFlow 1](ai_for_simulation/cosmoflow/tensorflow1)|
| et0 | Evapotransporation  | Inference | [TensorFlow 1](ai_for_simulation/et0/tensorflow1)  |
| Approximate Bayesian Computation (ABC) COVID-19 | Medical | Inference | [TensorFlow 2](ai_for_simulation/abc_covid_19/tensorflow2)  |

<br>

### <img width="25" src="https://user-images.githubusercontent.com/81682248/177360221-c599b6db-04e7-4e30-8be1-9752085df299.png"></a> Recommender Systems <a name="recommender_systems"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| Deep AutoEncoders for Collaborative Filtering | Recommender Systems | Training & Inference | [TensorFlow 1](recommendation/autoencoder/tensorflow1) |
| Click through rate: Deep Interest Network | Recommender Systems | Training & Inference | [TensorFlow 1](recommendation/click_through_rate/tensorflow1) |
<br>

### <img width="30" src="https://user-images.githubusercontent.com/81682248/177372896-6699b42b-c7ff-4186-93b9-b7f46ec03999.png"></a> Reinforcement Learning <a name="rl"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| RL Policy model | Reinforcement Learning | Training | [TensorFlow 1](reinforcement_learning/rl_policy_model/tensorflow1) |

<br>

### <img width="25" src="https://user-images.githubusercontent.com/81682248/177373761-77d40785-5390-400b-ad9f-305f4fd54a05.png"></a> Sparsity <a name="sparsity"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| MNIST RigL | Dynamic Sparsity | Training | [TensorFlow 1](sparsity/dynamic_sparsity/tensorflow1/mnist_rigl) |
| Autoregressive Language Modelling | Dynamic Sparsity | Training | [TensorFlow 1](sparsity/dynamic_sparsity/tensorflow1/language_modelling) 
| Block-Sparse library  | Sparsity | Training & Inference | [PopART](sparsity/block_sparse/popart) , [TensorFlow 1](sparsity/block_sparse/tensorflow1)|


<br>

### <img width="30" src="https://user-images.githubusercontent.com/81682248/177374313-c567fa25-c1a0-450f-855b-ce8b243d087e.png"></a> Probability <a name="probability"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| Contrastive Divergence VAE using MCMC methods  | Generative Model | Training | [TensorFlow 1](probability/contrastive_divergence_vae/tensorflow1) |
| mcmc  | Statistics | Training & Inference | [TensorFlow 1](probability/mcmc/tensorflow1/)|

<br>

### Miscellaneous <a name="miscellaneous"></a>
| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| Sales forecasting | MLP (Multi-Layer Perceptron) | Training | [TensorFlow 1](miscellaneous/sales_forecasting/tensorflow1) |
| Monte Carlo Ray Tracing  | Graphics | Inference | [Poplar](miscellaneous/monte_carlo_ray_tracing/poplar) |

<br>

### Archived <a name="archived"></a>

The following applications have been archived. More information can be provided on request.
| Model | Domain | Type | Framework|
| ------- | ------- |------- | ------- |
| Minigo | Reinforcement Learning | Training | TensorFlow 1|


<br> 

## Developer Resources
- [Documentation](https://docs.graphcore.ai/en/latest/): Explore our software documentation, user guides, and technical notes
- [Tutorials](https://github.com/graphcore/tutorials/tree/master/tutorials): Hands-on code tutorials, simple application and feature examples
- [How-to Videos](https://www.graphcore.ai/resources/how-to-videos): Watch practical how-to videos and demos by Graphcore engineers
- [Research Papers](https://www.graphcore.ai/resources/research-papers): Read publications from Graphcore's Research team and IPU innovators

<br>


## PopVision™ Tools
Visualise your code's inner workings with a user-friendly, graphical interface to optimise your machine learning models.

[Download](https://www.graphcore.ai/developer/popvision-tools) PopVision to analyse IPU performance and utilisation. 

<br>

## Support
Please note we are not currently accepting pull requests or issues on this repository. If you are actively using this repository and want to report any issues, please raise a ticket through the [Graphcore support portal](https://support.graphcore.ai/).

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
    *  Speech : FastSpeech2(TensorFlow 2), Transfomer Transducer(PopART), Conformer-Small(PyTorch)
    *  GNN : TGN(TensorFlow 1), MPNN(TensorFlow 2)
</details>

<br>
## Connect with us
<p align="center">
  <a href="https://twitter.com/graphcoreai"><img src="https://img.shields.io/badge/Twitter-1DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white"/></a>
  <a href="https://www.linkedin.com/company/graphcore"><img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white"/></a>
  <a href="http://www.facebook.com/pages/Graphcore/890447934394683"><img src="https://img.shields.io/badge/Facebook-1877F2.svg?style=for-the-badge&logo=Facebook&logoColor=white"/></a>
  <a href="https://www.youtube.com/c/Graphcore"><img src="https://img.shields.io/badge/YouTube-FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white"/></a>
  <a href="https://medium.com/graphcore"><img src="https://img.shields.io/badge/Medium-000000.svg?style=for-the-badge&logo=Medium&logoColor=white"/></a>
</p>
