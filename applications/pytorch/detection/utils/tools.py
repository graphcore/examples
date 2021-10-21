# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import collections
import numpy as np
from ruamel import yaml
import torchvision
import time
from typing import Dict, List, Tuple, Union
from yacs.config import CfgNode

import torch

from utils.visualization import scale_boxes_to_orig


class StatRecorder:
    """
    Records and prints the time stats and metrics of a model (latency and throughput)
    """
    def __init__(self, cfg: CfgNode):
        """
        Collecting latency and evaluation stats for inference and post processing
        Parameters:
            cfg: yacs object containing the config
            class_names: list of all the class names
        """
        self.total_times = []
        self.inference_times = []
        self.nms_times = []
        self.inference_throughputs = []
        self.total_throughputs = []
        self.eval_stats = []
        self.image_count = cfg.model.micro_batch_size * cfg.ipuopts.batches_per_step
        self.cfg = cfg
        self.class_names = yaml.safe_load(open(cfg.model.class_name_path))['class_names']
        self.seen = 0
        self.iou_values = torch.linspace(0.5, 0.95, 10)
        self.num_ious = self.iou_values.numel()

    def record_eval_stats(self, labels: np.array, predictions: np.array, image_size: torch.Tensor):
        """
        Records the statistics needed to compute the metrics
        Parameters:
            labels (np.array): N X 5 array of labels
            predictions (np.array): M X 85 array of predictions
            image_size (torch.Tensor): contains the original image size
        """
        labels = torch.from_numpy(labels)
        predictions = torch.from_numpy(predictions)

        num_labels = len(labels)
        target_cls = labels[:, 0].tolist() if num_labels else []
        self.seen = self.seen + 1

        # Handle case where we get no predictions
        if predictions is None:
            if num_labels:
                self.eval_stats.append((torch.zeros(0, self.num_ious, dtype=torch.bool), torch.Tensor(), torch.Tensor(), target_cls))
        else:
            bboxes = xywh_to_xyxy(predictions[:, :4])
            scores = predictions[:, 4]
            class_pred = predictions[:, 5]

            clip_coords(bboxes, image_size)

            # Assign all predictions as inccorect
            correct = torch.zeros(predictions.shape[0], self.num_ious, dtype=torch.bool)

            if num_labels:
                detected = []
                target_cls_tensor = labels[:, 0]

                # target boxes
                target_box = xywh_to_xyxy(labels[:, 1:])

                # Per target class
                for cls in torch.unique(target_cls_tensor):
                    target_indx = (cls == target_cls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pred_indx = (cls == predictions[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pred_indx.shape[0]:
                        # Prediction to target ious
                        best_ious, best_indxs = iou(predictions[pred_indx, :4], target_box[target_indx]).max(1)  # best ious, indices
                        # Appended detections
                        detected_set = set()
                        for iou_indx in (best_ious > self.iou_values[0]).nonzero(as_tuple=False):
                            detected_target = target_indx[best_indxs[iou_indx]]
                            if detected_target.item() not in detected_set:
                                detected_set.add(detected_target.item())
                                detected.append(detected_target)
                                correct[pred_indx[iou_indx]] = best_ious[iou_indx] > self.iou_values  # iou_thres is 1xn
                                if len(detected) == num_labels:  # all targets already located in image
                                    break

            self.eval_stats.append((correct.cpu(), scores.cpu(), class_pred.cpu(), target_cls))

    def compute_and_print_eval_metrics(self):
        """
        Computes and prints the evaluation metrics
        """
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        precision, recall, f1, mean_precision, mean_recall, map50, map = 0., 0., 0., 0., 0., 0., 0.
        ap = []
        eval_stats = [np.concatenate(x, 0) for x in zip(*self.eval_stats)]
        if len(eval_stats) and eval_stats[0].any():
            precision, recall, ap, f1, ap_class = ap_per_class(*eval_stats)
            precision, recall, ap50, ap = precision[:, 0], recall[:, 0], ap[:, 0], ap.mean(1)
            mean_precision, mean_recall, map50, map = precision.mean(), recall.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(eval_stats[3].astype(np.int64), minlength=len(self.class_names))  # number of targets per class
        else:
            nt = np.zeros(1)

        pf = '%20s' + '%12.5g' * 6  # print format
        print("\n EVALUTAION \n")
        print(s)
        print(pf % ('all', self.seen, nt.sum(), mean_precision, mean_recall, map50, map))
        if self.cfg.eval.verbose:
            for indx, cls in enumerate(ap_class):
                print(pf % (self.class_names[cls], self.seen, nt[cls], precision[indx], recall[indx], ap50[indx], ap[indx]))

    def record_inference_stats(self, nms_step_time: float, inference_round_trip_time: Tuple[float, float, float], inference_step_time: float):
        """Storages in the class the latency, inference time and postprocessing time of a model
            Parameters:
                inference_round_trip_time (Tuple): Latency tuple with (Min, Max, Avg) Latencies. This is the round trip time from host -> device -> host for a single batch
                inference_step_time (float): Inference time of a step
                nms_step_time (float): Postprocessing time of a step
        """

        # inference_round_trip_time is an average time needed for a step
        self.inference_times.append(inference_round_trip_time)
        # inference_step_time is the time taken to complete the step, and used to calculate the throughput
        inference_throughput = self.image_count/inference_step_time
        self.inference_throughputs.append(inference_throughput)

        self.nms_times.append(nms_step_time)

        total_step_time = inference_step_time + nms_step_time
        self.total_times.append(total_step_time)

        total_throughput = self.image_count/total_step_time
        self.total_throughputs.append(total_throughput)

    def logging(self, function):
        """Prints using the "function" given the average of the times recorded during the call to record_inference_stats()
            Parameters:
                function (function): function used to print the stats recorded
        """
        avg_nms_time_per_step = sum(self.nms_times)/len(self.nms_times)
        avg_total_time_per_step = sum(self.total_times)/len(self.total_times)

        avg_min_latency = [x[0] for x in self.inference_times]
        avg_max_latency = [x[1] for x in self.inference_times]
        avg_latency = [x[2] for x in self.inference_times]

        function("Inference stats: image size {}x{}, batches per step {}, batch size {}, {} steps".format(
            self.cfg.model.image_size, self.cfg.model.image_size, self.cfg.ipuopts.batches_per_step, self.cfg.model.micro_batch_size, len(self.total_times)
        ))
        function("--------------------------------------------------")
        function("Inference")
        function("Average Min Latency per Batch: {:.3f} ms".format(1000 * sum(avg_min_latency)/len(self.inference_times)))
        function("Average Max Latency per Batch: {:.3f} ms".format(1000 * sum(avg_max_latency)/len(self.inference_times)))
        function("Average Latency per Batch: {:.3f} ms".format(1000 * sum(avg_latency)/len(self.inference_times)))
        function("Average Inference Throughput: {:.3f} img/s".format(sum(self.inference_throughputs)/len(self.inference_throughputs)))
        function("--------------------------------------------------")
        # TODO remove the NMS and end-to-end time report once NMS is on device
        function("End-to-end")
        function("Average NMS Latency per Batch: {:.3f} ms".format(1000 * avg_nms_time_per_step/self.cfg.ipuopts.batches_per_step))
        function("Average End-to-end Latency per Batch: {:.3f} ms".format(1000 * avg_total_time_per_step/self.cfg.ipuopts.batches_per_step))
        function("End-to-end Throughput: {:.3f} img/s".format(sum(self.total_throughputs)/len(self.total_throughputs)))
        function("==================================================")

        if self.cfg.eval.metrics:
            self.compute_and_print_eval_metrics()


def load_weights(weight_path: str) -> List[Tuple[str, torch.Tensor]]:
    """
    Process the given state dict and return dictionary with matching layers
    Parameters:
        weight_path (str): the path to the weight file
    Returns:
        A list of layer names and weights for the model
    """
    model = torch.load(weight_path)
    model_weight = [(k, v) for (k, v) in model.items() if 'anchor' not in k]
    return model_weight


def map_key_names(scaled_yolo: List[Tuple[str, torch.Tensor]], our_model_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Translate scaled yolov4 weights' keys to our model weights' key
    Parameters:
        scaled_yolo (List[Tuple[str, torch.Tensor]]): Original weight state dict
        our_model_dict (Dict[str, torch.Tensor]): Our model state dict
    Returns:
        A model state dict with the original weights copied into our model
    """
    scaled_yolo_final = collections.OrderedDict()
    for i, key in enumerate(our_model_dict):
        if i < len(scaled_yolo):
            (k, v) = scaled_yolo[i]
            scaled_yolo_final[key] = v
    return scaled_yolo_final


def load_and_fuse_pretrained_weights(model: torch.nn.Module, opt: argparse.ArgumentParser) -> torch.nn.Module:
    """
    Given a model, load pretrained weight and fuse conv layers
    Parameters:
        model (torch.nn.Module): Placeholder model to load weights into
        opt (argparse.ArgumentParser): Object that contains the path to the model weights
    Returns:
        A model fused with the weights
    """
    model.optimize_for_inference(fuse_all_layers=False)
    fused_state_dict = load_weights(opt.weights)
    fused_state_dict = map_key_names(fused_state_dict, model.state_dict())
    model.load_state_dict(fused_state_dict)
    return model


def intersection(boxes1: np.array, boxes2: np.array) -> np.array:
    """
    Find the areas of intersection between the given boxes1 and boxes2 in xmin, ymin, xmax, ymax
    Parameters:
        boxes1 (np.array): NX4 array of boxes
        boxes2 (np.array): MX4 array of boxes
    Returns:
        A NXM array of intersection of boxes
    """
    inter_coords = np.clip(
        np.minimum(boxes1[:, None, 2:], boxes2[:, 2:]) - np.maximum(boxes1[:, None, :2], boxes2[:, :2]),
        a_min=0,
        a_max=None
    )

    return np.prod(inter_coords, 2)


def area(boxes: Union[np.array, torch.Tensor]) -> Union[np.array, torch.Tensor]:
    """
    Find the areas of the given boxes in xmin, ymin, xmax, ymax
    Parameters:
        boxes (np.array): NX4 array of boxes
    Returns:
        A NX1 array of box areas
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def ioa(boxes1: np.array, boxes2: np.array) -> np.array:
    """
    Compute the are of intersection between boxes1 and boxes2
        over the area of boxes2
    Parameters:
        boxes1 (np.array): M x 4 representing xmin, ymin, xmax, ymax of bounding box
        boxes2 (np.array): N x 4 representing xmin, ymin, xmax, ymax of bounding box
    Returns:
         (M x N) IoA between boxes1 and boxes2.
    """

    inter = intersection(boxes1, boxes2)
    area2 = area(boxes2)

    return inter / area2


def iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Return intersection-over-union of boxes.
    Both sets of boxes are expected to be in (xmin, ymin, xmax, ymax) format.
    Arguments:
        boxes1 (torch.Tensor): a NX4 tensor of boxes
        boxes2 (torch.Tensor): a MX4 tensor of boxes
    Returns:
        torch.Tensor: the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = area(boxes1)
    area2 = area(boxes2)

    inter = (torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def nms(predictions: torch.Tensor, iou_threshold: float, score_threshold: float, max_detections: int = 300) -> List[torch.Tensor]:
    """
    Perform non maximum suppression on predictions.
    Parameters:
        predictions (torch.Tensor): Nx85 representing xmin, ymin, xmax, ymax, obj_scores, cls_scores
        iou_threshold (float):  Predictions that overlap by more than this threshold will be discarded
        score_threshold (float): Predictions less than this prob will be discarded
        max_detections (int) : Maximum number of detections per image
    Returns:
        List[torch.Tensor]: Mx6 predictions filtered after NMS
    """
    max_image_dimension = 4096
    output = [None] * predictions.shape[0]

    valid_predictions_indices = predictions[..., 4] > score_threshold  # indices for predictions that are above the threshold.

    # iterate over batches
    for pred_indx, pred in enumerate(predictions):
        pred = pred[valid_predictions_indices[pred_indx]]
        if not pred.shape[0]:
            continue

        # Compute confidence
        pred[:, 5:] = pred[:, 5:] * pred[:, 4:5]
        boxes = xywh_to_xyxy(pred[:, :4])

        # Multi-Class NMS
        # Select classes for which confidence is above the threshold.
        box_indx, class_indx = (pred[:, 5:] > score_threshold).nonzero(as_tuple=False).T  # Class indices that exceed threshold.
        pred = torch.cat((boxes[box_indx], pred[box_indx, class_indx + 5, None], class_indx[:, None].float()), 1)  # Assign valid classes to the bboxes.

        if not pred.shape[0]:
            continue

        box_shift = pred[:, 5:6] * (max_image_dimension)
        shifted_box, scores = pred[:, :4] + box_shift, pred[:, 4]
        selected_box_indx = torchvision.ops.boxes.nms(shifted_box, scores, iou_threshold)
        if selected_box_indx.shape[0] > max_detections:  # limit detections
            selected_box_indx = selected_box_indx[:max_detections]
        output[pred_indx] = pred[selected_box_indx]

    return output


def post_processing(
    cfg: CfgNode, y: torch.Tensor, orig_img_size: torch.Tensor, transformed_labels: torch.Tensor
) -> Tuple[Tuple[List[np.array], List[np.array]], float]:
    """
    Post process raw prediction output from the model, and convert the normalized label to image size scale
    Parameters:
        cfg: yacs object containing the config
        y (torch.Tensor): tensor output from the model
        orig_img_size (torch.Tensor):  size of the original image (h, w)
        transformed_labels (torch.Tensor): labels from dataloader, that has been transformed with augmentation if applicable
    Returns:
        Tuple[Tuple[List[np.array], List[np.array]], float]: a tuple of processed predictions, processed labels
        and the time taken to compute the post-processing
    """
    post_processing_start_time = time.time()
    pruned_preds_batch = post_process_prediction(y, orig_img_size, cfg)
    post_processing_end_time = time.time()
    processed_labels_batch = post_process_labels(transformed_labels, orig_img_size, cfg)

    return (pruned_preds_batch, processed_labels_batch), (post_processing_end_time - post_processing_start_time)



def post_process_prediction(y: torch.Tensor, orig_img_sizes: torch.Tensor, cfg: CfgNode) -> List[np.array]:
    """
    Post process the raw output from the model, including filtering predictions
        with higher probability than the object score, apply nms and scale the prediction back
        to the original image size
    Parameters:
        y (torch.Tensor): tensor output from the model
        orig_img_size (torch.Tensor):  size of the original image (h, w)
        cfg: yacs object containing the config
    Returns:
        List[np.array]: an array of processed bounding boxes, associated class score
        and class prediction of each bounding box
    """
    output = torch.cat(y, axis=1).float()
    predictions = nms(output, cfg.inference.iou_threshold, cfg.inference.class_conf_threshold)
    scaled_preds = []
    for i, pred in enumerate(predictions):
        if pred is None:
            continue
        pred = pred.detach().numpy()
        pred = xyxy_to_xywh(pred)
        orig_img_size = orig_img_sizes[i]
        scaled_pred = scale_boxes_to_orig(pred, orig_img_size[1], orig_img_size[0], cfg.model.image_size)
        scaled_preds.append(scaled_pred)
    return scaled_preds


def post_process_labels(labels: torch.Tensor, orig_img_sizes: torch.Tensor, cfg: CfgNode) -> List[np.array]:
    """
    Post process raw prediction output from the model, and convert the normalized label to image size scale
    Parameters:
        labels (torch.Tensor): labels from dataloader, that has been transformed with augmentation if applicable
        orig_img_sizes (torch.Tensor):  size of the original image (h, w)
        cfg: yacs object containing the config
    Returns:
        List[np.array]: a list of processed labels
    """

    labels = labels.detach().numpy()

    processed_labels = []
    for i, label in enumerate(labels):
        # Remove label padding
        label = label[np.abs(label.sum(axis=1)) != 0.]
        label = standardize_labels(label, cfg.model.image_size, cfg.model.image_size)
        orig_img_size = orig_img_sizes[i]
        scaled_boxes = scale_boxes_to_orig(label[:, 1:], orig_img_size[1], orig_img_size[0], cfg.model.image_size)
        label[:, 1:] = scaled_boxes
        processed_labels.append(label)
    return processed_labels


def standardize_labels(labels: np.array, width: int, height: int) -> np.array:
    """
    Convert normalized label (as the ratio of image size) to the pixel scale of the image size.
    Parameters:
        labels (np.array): NX5 array of labels
        width (int): image width
        height (int): image height
    Returns:
        (np.array): an NX5 array of standardized labels
    """
    labels[:, [1, 3]] *= width
    labels[:, [2, 4]] *= height
    return labels


def xyxy_to_xywh(boxes: np.array) -> np.array:
    """
    Convert xmin, ymin, xmax, ymax to centerx, centery, width, height
    Parameters:
        boxes (np.array): boxes in the xmin, ymin, xmax, ymax format
    Returns:
        np.array: boxes in the centerx, centery, width, height format
    """
    boxes[..., 2] = boxes[..., 2] - boxes[..., 0]
    boxes[..., 3] = boxes[..., 3] - boxes[..., 1]
    boxes[..., 0] = boxes[..., 0] + boxes[..., 2]/2
    boxes[..., 1] = boxes[..., 1] + boxes[..., 3]/2
    return boxes


def xywh_to_xyxy(boxes: np.array) -> np.array:
    """
    Convert centerx, centery, width, height to xmin, ymin, xmax, ymax
    Parameters:
        boxes (np.array): boxes in the centerx, centery, width, height format
    Returns:
        np.array: boxes in the xmin, ymin, xmax, ymax format
    """
    boxes[..., 0] = boxes[..., 0] - boxes[..., 2]/2
    boxes[..., 1] = boxes[..., 1] - boxes[..., 3]/2
    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    return boxes


def ap_per_class(
    tp: np.array, conf: np.array, pred_cls: np.array, target_cls: np.array
) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Compute the precision-recall curve and the average precision
    Parameters:
        tp (np.array): True positives (NX1 or NX10)
        conf (np.array): Predicted confidence score from 0 to 1
        pred_cls (np.array): Predicted object class
        target_cls (np.array): Target object class
    Returns:
        Tuple[np.array, ...]: The precision, recall, average precision, f1 score, and unique classes
    """
    # Sort by confidence
    sorted_indices = np.argsort(-conf)
    tp, conf, pred_cls = tp[sorted_indices], conf[sorted_indices], pred_cls[sorted_indices]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create PR curve and compute AP metric for each class
    pr_score = 0.1
    metric_dim = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    average_precision, precision_array, recall_array = np.zeros(metric_dim), np.zeros(metric_dim), np.zeros(metric_dim)

    for cls_indx, cls in enumerate(unique_classes):
        pos_cls = pred_cls == cls
        num_gt = (target_cls == cls).sum()
        num_pos = pos_cls.sum()

        if num_pos == 0 or num_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fp_count = (1 - tp[pos_cls]).cumsum(0)
            tp_count = tp[pos_cls].cumsum(0)

            # Recall
            recall = tp_count / (num_gt + 1e-16)
            recall_array[cls_indx] = np.interp(-pr_score, -conf[pos_cls], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tp_count / (tp_count + fp_count)  # precision curve
            precision_array[cls_indx] = np.interp(-pr_score, -conf[pos_cls], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                average_precision[cls_indx, j] = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * precision_array * recall_array / (precision_array + recall_array + 1e-16)
    return (precision_array, recall_array, average_precision, f1, unique_classes.astype(int))


def compute_ap(recall: np.array, precision: np.array) -> np.array:
    """
    Compute the average precision, given the recall and precision curves
    Parameters:
        recall (np.array): The recall curve
        precision (np.array): The precision curve
    Returns:
        np.array: The average precision
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

    return ap


def clip_coords(boxes: torch.Tensor, img_shape: torch.Tensor):
    """
    Clip boxes to image dimensions
    Parameters
        boxes (torch.Tensor): a tensor of boxes
        img_shape (torch.Tensor): dimensions to clip at
    """
    boxes[:, 0].clamp_(0, img_shape[0])
    boxes[:, 1].clamp_(0, img_shape[1])
    boxes[:, 2].clamp_(0, img_shape[0])
    boxes[:, 3].clamp_(0, img_shape[1])
