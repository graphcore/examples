# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
from typing import Tuple, List
from yacs.config import CfgNode

import torch
import torch.nn as nn

from utils.custom_ops import Nms
from utils.tools import standardize_labels, xywh_to_xyxy, xyxy_to_xywh
from utils.visualization import scale_boxes_to_orig


class IPUPredictionsPostProcessing(nn.Module):
    def __init__(self, inference_cfg: CfgNode, cpu_mode: bool, testing_preprocessing: bool = False):
        super().__init__()
        self.score_threshold = inference_cfg.class_conf_threshold
        self.pre_nms_topk_k = inference_cfg.pre_nms_topk_k
        self.max_image_dimension = torch.tensor([4096.0])
        self.nms = Nms(inference_cfg, cpu_mode)
        self.cpu_mode = cpu_mode
        self.testing_preprocessing = testing_preprocessing

    def indexing_with_batch_size(self, x: torch.Tensor, index: torch.Tensor):
        """
        Indexes a 2 or 3 dimensional Tensor x, with the indices "index" in the last dimension,
           assuming the 0 dimension is the batch.
        Parameters:
            x (torch.Tensor): tensor to be indexed
            index (torch.Tensor): indices
        Returns:
            result (torch.Tensor): indexed result
        """
        if len(x.shape) == 3:
            # Just concat the batch dimension, then un-concat at the end
            batch_size = x.shape[0]
            batch_size_tensor = torch.arange(batch_size).unsqueeze(axis=-1)
            row_index = ((batch_size_tensor * x.shape[1]) + index).flatten()

            x = x.view(x.shape[0] * x.shape[1], x.shape[2])

            result = torch.index_select(x, 0, row_index)

            result = result.view(batch_size, index.shape[1], x.shape[-1])
            return result
        elif len(x.shape) == 2:
            return torch.gather(x, 1, index)

        raise ValueError("Wrong number of dimensions; expected 2 or 3.")

    def nms_preprocessing(self, predictions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Prepares the predictions to be processed by the NMS algorithm
        Parameters:
            predictions (torch.Tensor): predictions from the YOLOv4 inference model
        Returns:
            scores (torch.Tensor): scores of the predictions
            shifted_box (torch.Tensor): boxes of the predictions shifted by box_shift
            classes (torch.Tensor): classes of the predictions
            box_shift (torch.Tensor): classes * self.max_image_dimension to shift the boxes
        """
        batch_size = predictions.shape[0]
        n_classes = predictions.shape[-1] - 5

        # We get all the info from the predictions
        boxes = xywh_to_xyxy(predictions[..., :4])
        scores = predictions[..., 4]
        one_hot_encoded_classes = predictions[..., 5:]

        # We filter out most of values since they are going to be close to 0. or 0.
        valid_scores, valid_scores_indices = torch.topk(scores, k=self.pre_nms_topk_k)

        # we zero out all the values smaller than self.score_threshold
        valid_scores = valid_scores * (valid_scores > self.score_threshold)

        # We use the position of the filter for the predicted one hot encoded classes
        one_hot_encoded_classes = self.indexing_with_batch_size(one_hot_encoded_classes, valid_scores_indices)

        # We get new scores which are the multiplication of the class score with the objectness score
        multiplied_scores = (one_hot_encoded_classes * valid_scores.unsqueeze(axis=-1)).view(batch_size, -1)

        if self.cpu_mode and not self.testing_preprocessing:
            # We create indices for all the positions in multiplied scores
            all_indices = torch.arange(multiplied_scores.shape[1]).unsqueeze(axis=0).repeat(batch_size, 1)
            # We make the indices 0 where the multiplied scores are below self.score_threshold
            masked_indices = (all_indices * (multiplied_scores > self.score_threshold)).long()

            # From the masked indices the % n_classes will be the classes
            classes = masked_indices % n_classes
            # and // n_classes will be the box position
            box_indices = (masked_indices // n_classes).long()

            # We calculate the final box indices which will be the valid scores indices indexed by the box indices
            box_indices = self.indexing_with_batch_size(valid_scores_indices, box_indices)

            # We gather the boxes and scores
            boxes = self.indexing_with_batch_size(boxes, box_indices)
            scores = self.indexing_with_batch_size(multiplied_scores, masked_indices)

            # We create a shift so that the classes can be separated in space to make the job easier for the NMS
            box_shift = (classes * self.max_image_dimension).unsqueeze(axis=-1)
            shifted_box = boxes.float() + box_shift.float()

            return scores, shifted_box, classes.long()
        else:
            # We make the indices 0 where the multiplied scores are below self.score_threshold
            scores = multiplied_scores * (multiplied_scores > self.score_threshold)

            # We calculate the final box indices which will be the valid scores indices indexed by the box indices
            boxes = self.indexing_with_batch_size(boxes, valid_scores_indices)

            return scores.view(batch_size, -1, n_classes), boxes, 0

    def nms_postprocessing(self, scores: torch.Tensor, boxes: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """
        Prepares the predictions from NMS to the predictions returned by the model
        Parameters:
            scores (torch.Tensor): scores of the predictions from NMS
            boxes (torch.Tensor): boxes of the predictions from NMS
            classes (torch.Tensor): boxes of the predictions from NMS
        Returns:
            predictions (torch.Tensor): Predictions after NMS
        """
        if self.cpu_mode:
            # We de-shift the boxes so they are back to their original positions
            box_shift = (classes * self.max_image_dimension).unsqueeze(axis=-1)
            boxes = boxes - box_shift

        # We give final shape to out predictions after nms
        return torch.cat((boxes, scores.unsqueeze(axis=-1), classes.unsqueeze(axis=-1).float()), axis=-1)

    def forward(self, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process the predictions from the model using Non Maximum Suppression
        Parameters:
            predictions (torch.Tensor): predictions from the YOLOv4 inference model
        Returns:
            predictions (torch.Tensor): Predictions after NMS
            true_max_detections (torch.Tensor): int that points where the NMS detections finishes
        """
        scores_pre_nms, boxes_pre_nms, classes_pre_nms = self.nms_preprocessing(predictions)
        if self.testing_preprocessing:
            return scores_pre_nms, boxes_pre_nms
        _, scores, boxes, classes, true_max_detections = self.nms(
            scores_pre_nms.float(), boxes_pre_nms.float(), classes_pre_nms
        )
        predictions = self.nms_postprocessing(scores, boxes, classes)
        return predictions, true_max_detections


def post_processing(
    cfg: CfgNode, y: torch.Tensor, orig_img_size: torch.Tensor, transformed_labels: torch.Tensor
) -> Tuple[Tuple[List[np.array], List[np.array]], float]:
    """
    CPU post-processing function for the raw prediction output from the model, and convert the normalized label
        to image size scale
    Parameters:
        cfg: yacs object containing the config
        y (torch.Tensor): tensor output from the model
        orig_img_size (torch.Tensor):  size of the original image (h, w)
        transformed_labels (torch.Tensor): labels from dataloader, that has been transformed with augmentation if applicable
    Returns:
        Tuple[Tuple[List[np.array], List[np.array]], float]: a tuple of processed predictions, processed labels
        and the time taken to compute the post-processing
    """
    pruned_preds_batch = post_process_prediction(y, orig_img_size, cfg)
    processed_labels_batch = post_process_labels(transformed_labels, orig_img_size, cfg)

    return pruned_preds_batch, processed_labels_batch


def post_process_prediction(y: torch.Tensor, orig_img_sizes: torch.Tensor, cfg: CfgNode) -> List[np.array]:
    """
    CPU post-processing function for the raw output from the model. It scales the prediction back
        to the original image size
    Parameters:
        y (torch.Tensor): tensor output from the model
        orig_img_size (torch.Tensor):  size of the original image (h, w)
        cfg: yacs object containing the config
    Returns:
        List[np.array]: an array of processed bounding boxes, associated class score
        and class prediction of each bounding box
    """
    predictions, max_n_predictions = y
    scaled_preds = []
    for i, (pred, max_n) in enumerate(zip(predictions, max_n_predictions)):
        if pred is None or max_n == 0:
            continue
        pred = pred[:max_n]
        pred = xyxy_to_xywh(pred)
        scaled_pred = scale_boxes_to_orig(pred, orig_img_sizes[i], cfg.model.image_size)
        scaled_preds.append(scaled_pred)
    return scaled_preds


def post_process_labels(labels: torch.Tensor, orig_img_sizes: torch.Tensor, cfg: CfgNode) -> List[np.array]:
    """
    CPU post-processing function for the raw prediction output from the model. It converts the normalized label to
        image size scale
    Parameters:
        labels (torch.Tensor): labels from dataloader, that has been transformed with augmentation if applicable
        orig_img_sizes (torch.Tensor):  size of the original image (h, w)
        cfg: yacs object containing the config
    Returns:
        List[np.array]: a list of processed labels
    """
    processed_labels = []
    for i, label in enumerate(labels):
        # Remove label padding
        label = label[torch.abs(label.sum(axis=1)) != 0.0]
        label = standardize_labels(label, cfg.model.image_size, cfg.model.image_size)
        scaled_boxes = scale_boxes_to_orig(label[:, 1:], orig_img_sizes[i], cfg.model.image_size)
        label[:, 1:] = scaled_boxes
        processed_labels.append(label)
    return processed_labels
