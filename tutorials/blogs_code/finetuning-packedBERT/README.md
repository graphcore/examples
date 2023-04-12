# A deeper dive into faster fine-tuning with Packed BERT

This notebook is a walkthrough diving deeper into fine-tuning with a more online-capable version of PackedBERT, building on the original Packed BERT used for BERT pre-training. It explores all of the modifications made for preprocessing, dataset creation, the model itself and postprocessing required to implement packing for BERT (and similar models) using an example of sequence classification. The code is broken down and explained step by step, explaining the logic behind the choices made.

You can try the [notebooks](https://console.paperspace.com/github/gradient-ai/Graphcore-HuggingFace?file=%2Fpacked-bert) that use these functions in Optimum Graphcore in Paperspace.

Before starting the walkthrough, ensure you have all requirements installed with

```
pip install -r requirements.txt
```
