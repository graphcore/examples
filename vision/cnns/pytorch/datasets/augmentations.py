# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import torch


class AugmentationModel(torch.nn.Module):
    def __init__(self, model, use_mixup, use_cutmix, args=None):
        super().__init__()
        self.model = model
        assert use_mixup or use_cutmix, (
            "AugmentationModel needs to use at least one "
            "augmentation technique.")
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        if self.use_mixup:
            assert args is not None and\
                hasattr(args, 'mixup_alpha'), (
                    "If using mixup, mixup_alpha must be defined."
                )
            self.mixup_alpha = args.mixup_alpha

        if self.use_cutmix:
            assert args is not None and\
                hasattr(args, 'cutmix_lambda_low') and\
                hasattr(args, 'cutmix_lambda_high') and\
                hasattr(args, 'cutmix_disable_prob'), (
                    "If using cutmix, cutmix parameters (lambda_low/high "
                    "and disable_prob) must be passed to the AugmentationModel."
                )
            if args.cutmix_lambda_low == args.cutmix_lambda_high:
                self._sample_cutmix = torch.tensor(args.cutmix_lambda_low)
            else:
                self._sample_cutmix = torch.distributions.Uniform(
                    args.cutmix_lambda_low,
                    args.cutmix_lambda_high,
                )
            self._cutmix_disable_prob = args.cutmix_disable_prob
        self._sampling_in_half = args.precision[:3] == "16."

    def forward(self, batch):
        coeffs = []
        if self.use_mixup:
            mixup_coeffs = self._sample_mixup_coefficients(batch.size()[0])
            coeffs.append(mixup_coeffs)
            batch = self._mixup(batch, mixup_coeffs)

        if self.use_cutmix:
            cutmix_coeff = self._sample_cutmix_coeff()
            batch, cutmix_coeff = self._cutmix(batch, cutmix_coeff)
            coeffs.append(cutmix_coeff)

        return self.model(batch), coeffs

    def _mixup(self, batch, coeffs):
        coeffs = coeffs.reshape((coeffs.shape[0], 1, 1, 1))
        return coeffs * batch + (1.0 - coeffs) * self._permute(batch, 1)

    def _cutmix(self, batch, coeff):
        height, width = batch.shape[2], batch.shape[3]
        cut_width = width * torch.sqrt(1.0 - coeff)
        cut_height = height * torch.sqrt(1.0 - coeff)

        # Make sure the cut box doesn't reach over the original image,
        # i.e. sample uniformly in [cut_dim / 2, dim - cut_dim / 2).
        cut_center_x = (width - cut_width) * torch.rand((1,)) + (cut_width / 2.0)
        cut_center_y = (height - cut_height) * torch.rand((1,)) + (cut_height / 2.0)

        # Final cut box coordinates.
        cut_x1 = torch.round(cut_center_x - cut_width / 2.0).to(torch.int32)
        cut_x2 = torch.round(cut_center_x + cut_width / 2.0).to(torch.int32)
        cut_y1 = torch.round(cut_center_y - cut_height / 2.0).to(torch.int32)
        cut_y2 = torch.round(cut_center_y + cut_height / 2.0).to(torch.int32)

        # Create the cut box mask.
        xs = torch.arange(width)
        ys = torch.arange(height)
        xs_mask = (xs >= cut_x1).to(torch.int32) * (xs < cut_x2).to(torch.int32)
        ys_mask = (ys >= cut_y1).to(torch.int32) * (ys < cut_y2).to(torch.int32)
        mask = (xs_mask * ys_mask.reshape((height, 1)))

        coeff = 1.0 - torch.mean(mask.to(torch.float32))
        batch_cutmixed = torch.where(mask.to(torch.bool), self._permute(batch, 2), batch)
        return batch_cutmixed, coeff


    def _sample_mixup_coefficients(self, batch_size):
        # This sampling method work well only for alpha <= 1.0
        dtype = torch.float16 if self._sampling_in_half else torch.float32
        assert self.mixup_alpha >= 0, "Beta distribution is not implemented for alpha < 0.0"
        assert self.mixup_alpha <= 1.0, "Beta distribution is not implemented for alpha > 1.0"

        def generate_rand():
            return torch.rand(batch_size, dtype=dtype)

        # clip between (eps, 1-eps)
        def clip(tensor):
            eps = torch.tensor([torch.finfo(dtype).eps]).type(dtype)
            tensor = torch.max(eps, tensor)
            tensor = torch.min(1.0 - eps, tensor)
            return tensor

        U = generate_rand()
        V = generate_rand()
        X = clip(torch.pow(U, 1.0 / self.mixup_alpha))
        Y = clip(torch.pow(V, 1.0 / self.mixup_alpha))
        X2 = clip(torch.pow((1.0 - U), 1.0 / self.mixup_alpha))
        Y2 = clip(torch.pow((1.0 - V), 1.0 / self.mixup_alpha))
        XpY = X + Y
        XpY2 = X2 + Y2
        result = torch.where((XpY <= torch.ones(1, dtype=dtype)) & (XpY > torch.zeros(1, dtype=dtype)), X / XpY, X2 / XpY2)
        # Original image is the foreground image (i.e. coefficient >= 0.5),
        # each image is once an original image and once a target image
        # within the batch.
        return torch.max(result, 1.0 - result)


    def _sample_cutmix_coeff(self):
        # Cutmix coefficient is the same for all images within a batch and is
        # generated on the device.
        if isinstance(self._sample_cutmix, torch.Tensor):
            coeff = self._sample_cutmix
        else:
            coeff = self._sample_cutmix.sample()
        disabled_coeff = torch.tensor(1.0)
        coeff = coeff.to(disabled_coeff.device)

        # Maybe disable cutmix.
        coeff = torch.where(
            torch.rand(()) < self._cutmix_disable_prob,
            # Cutmix disabled.
            disabled_coeff,
            # Cutmix enabled.
            coeff,
        )

        if self._sampling_in_half:
            return coeff.half()
        return coeff

    def mix_labels(self, labels, coeffs):
        if self.use_mixup and self.use_cutmix:
            mixup_coeffs = coeffs[0].to(torch.float32)
            permuted_mixup_coeffs = self._permute(mixup_coeffs, 2)
            cutmix_coeffs = coeffs[1].to(torch.float32)

            # The resulting image is a combination of 4 images. Two images from
            # cutmix, which are in turn each a combination of 2 images from
            # mixup.
            all_labels = [
                labels,
                self._permute(labels, 1),
                self._permute(labels, 2),
                self._permute(labels, 3),
            ]
            weights = [
                mixup_coeffs * cutmix_coeffs,
                (1.0 - mixup_coeffs) * cutmix_coeffs,
                permuted_mixup_coeffs * (1.0 - cutmix_coeffs),
                (1.0 - permuted_mixup_coeffs) * (1.0 - cutmix_coeffs),
            ]

            return all_labels, weights

        if self.use_mixup:
            mixup_coeffs = coeffs[0].to(torch.float32)
            all_labels = [labels, self._permute(labels, 1)]
            weights = [mixup_coeffs, (1.0 - mixup_coeffs)]
            return all_labels, weights

        if self.use_cutmix:
            cutmix_coeffs = coeffs[0].to(torch.float32)
            all_labels = [labels, self._permute(labels, 2)]
            weights = [cutmix_coeffs, (1.0 - cutmix_coeffs)]
            return all_labels, weights

        raise ValueError(
            "AugmentationModel needs to use at least one "
            "augmentation technique.")


    @staticmethod
    def _permute(tensor, shifts):
        return torch.roll(tensor, shifts=shifts, dims=0)
