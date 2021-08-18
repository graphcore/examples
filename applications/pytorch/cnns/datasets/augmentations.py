# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import torch


# We sample mixup coefficients on the host.
def sample_mixup_coefficients(alpha, batch_size, np_type, random_generator):
    coefficients = random_generator.beta(alpha, alpha, size=batch_size)
    coefficients = coefficients.astype(np.float32, copy=False)
    # Original image is the foreground image (i.e. coefficient >= 0.5),
    # each image is once an original image and once a target image
    # within the batch.
    coefficients = np.maximum(coefficients, 1.0 - coefficients)
    return torch.from_numpy(coefficients.astype(np_type, copy=False))


# Mixup is done on the device.
class MixupModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch_and_mixup_coefficients, labels):
        batch, coeffs = batch_and_mixup_coefficients
        coeffs = coeffs.reshape((coeffs.shape[0], 1, 1, 1))
        batch_augmented = coeffs * batch + (1.0 - coeffs) * self._permute(batch)
        # We return permuted labels instead of augmented labels because
        # the wrapper model needs them in order to compute the loss.
        return self.model(batch_augmented), self._permute(labels)

    @staticmethod
    def _permute(items):
        return torch.roll(items, shifts=1, dims=0)
