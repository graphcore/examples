# Code examples for TensorFlow 2

This directory contains several examples showing how to use TensorFlow 2 on the IPU.

- [Adversarial Generalized Method of Moments](adversarial_generalized_method_of_moments): This example is an implementation of Adversarial Generalized Method of Moments, an approach for solving statistical problems based on generative adversarial networks with a wide variety of applications.

- [CIFAR-10 with IPUEstimator](cifar10): This example shows how to train a model to sort images from the CIFAR-10 dataset using the IPU implementation of the TensorFlow Estimator API.

- [Graph Neural Network Example](gnn): This example uses the Spektral GNN library to predict the heat capacity of various molecules in the QM9 dataset.

- [IMDB Sentiment Prediction](imdb): These examples train an IPU model with an embedding layer and an LSTM to predict the sentiment of an IMDB review.

- [Inspecting tensors using custom outfeed layers and a custom optimizer](inspecting_tensors): This example trains a choice of simple fully connected models on the MNIST numeral data set and shows how tensors (containing activations and gradients) can be returned to the host via outfeeds for inspection.

- [Simple MNIST training example](mnist): This example trains a simple 2-layer fully connected model on the MNIST numeral data set.

- [Shakespeare corpus reader](shakespeare): This example learns to predict the next character in the corpus of William Shakespeare.
