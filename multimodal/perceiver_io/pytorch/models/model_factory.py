# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from dataclasses import asdict

from transformers import PerceiverConfig, PerceiverFeatureExtractor

from configs.hparams import AVAILABLE_MODELS, ModelArguments
from models.modelling_perceiver import register_subclass


def get_model(model_args: ModelArguments):
    model_cls = AVAILABLE_MODELS[model_args.model_name]
    register_subclass(model_cls)
    # update the base config
    model_config = PerceiverConfig.from_pretrained(model_args.model_name)
    model_config.update(asdict(model_args))
    # get feature extractor
    feature_extractor = PerceiverFeatureExtractor.from_pretrained(model_args.model_name)
    # instantiate the model based on the config
    model = model_cls(config=model_config)
    return model, feature_extractor
