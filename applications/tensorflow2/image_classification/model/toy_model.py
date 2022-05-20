# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from tensorflow import keras
import logging

logger = logging.getLogger('toy_model')


def ToyModel(input_shape, classes=1000):

    logger.info(f'creating a ToyModel with {classes} classes and input shape {input_shape}')

    img_input = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(filters=4, kernel_size=3, activation='relu', name='conv2d_0')(img_input)
    x = keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5))(x)
    x = keras.layers.Conv2D(filters=8, kernel_size=1, activation='relu', name='conv2d_1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5))(x)
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(classes, activation='softmax')(x)

    return keras.Model(img_input, x, name='toy_model')


def ToyModelBn(input_shape, classes=1000):

    logger.info(f'creating a ToyModelBN with {classes} classes and input shape {input_shape}')

    img_input = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(filters=4, kernel_size=3, activation='relu', name='conv2d_0')(img_input)
    x = keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5))(x)
    x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=f'bn_0')(x)
    x = keras.layers.Conv2D(filters=8, kernel_size=1, activation='relu', name='conv2d_1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5))(x)
    x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=f'bn_1')(x)
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(classes, activation='softmax')(x)

    return keras.Model(img_input, x, name='toy_model')
