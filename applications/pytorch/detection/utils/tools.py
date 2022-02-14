# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import collections
import numpy as np
import os
from ruamel import yaml
from scipy.cluster.vq import kmeans
import torchvision
import time
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
from yacs.config import CfgNode

import torch
from torchvision.ops.boxes import nms as torchvision_nms

from utils.anchors import AnchorBoxes
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
        self.class_names = yaml.safe_load(open(os.environ['PYTORCH_APPS_DETECTION_PATH'] + "/" + cfg.model.class_name_path))['class_names']
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
                        best_ious, best_indxs = iou(bboxes[pred_indx, :4], target_box[target_indx]).max(1)  # best ious, indices
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

            self.eval_stats.append((correct.cpu().detach().numpy(), scores.cpu().detach().numpy(), class_pred.cpu().detach().numpy(), target_cls))

    def compute_and_print_eval_metrics(self, output_function):
        """
        Computes and prints the evaluation metrics
        """
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        precision, recall, f1, mean_precision, mean_recall, m_ap50, m_ap = 0., 0., 0., 0., 0., 0., 0.
        ap = []
        eval_stats = [np.concatenate(x, 0) for x in zip(*self.eval_stats)]
        if len(eval_stats) and eval_stats[0].any():
            precision, recall, ap, f1, ap_class = ap_per_class(*eval_stats)
            precision, recall, ap50, ap = precision[:, 0], recall[:, 0], ap[:, 0], ap.mean(1)
            mean_precision, mean_recall, m_ap50, m_ap = precision.mean(), recall.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(eval_stats[3].astype(np.int64), minlength=len(self.class_names))  # number of targets per class
        else:
            nt = np.zeros(1)

        pf = '%20s' + '%12.5g' * 6  # print format
        output_function("\n EVALUATION \n")
        output_function(s)
        output_function(pf % ('all', self.seen, nt.sum(), mean_precision, mean_recall, m_ap50, m_ap))
        if self.cfg.eval.verbose:
            for indx, cls in enumerate(ap_class):
                output_function(pf % (self.class_names[cls], self.seen, nt[cls], precision[indx], recall[indx], ap50[indx], ap[indx]))

        return self.seen, nt.sum(), mean_precision, mean_recall, m_ap50, m_ap

    def record_inference_stats(self, inference_round_trip_time: Tuple[float, float, float], inference_step_time: float):
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

    def logging(self, output_function):
        """Prints using the "output_function" given the average of the times recorded during the call to record_inference_stats()
            Parameters:
                output_function (function): function used to print the stats recorded
        """
        avg_min_latency = [x[0] for x in self.inference_times]
        avg_max_latency = [x[1] for x in self.inference_times]
        avg_latency = [x[2] for x in self.inference_times]

        output_function("Inference stats: image size {}x{}, batches per step {}, batch size {}, {} steps".format(
            self.cfg.model.image_size, self.cfg.model.image_size, self.cfg.ipuopts.batches_per_step, self.cfg.model.micro_batch_size, len(self.total_times)
        ))
        output_function("--------------------------------------------------")
        output_function("Inference")
        output_function("Average Min Latency per Batch: {:.3f} ms".format(1000 * sum(avg_min_latency)/len(self.inference_times)))
        output_function("Average Max Latency per Batch: {:.3f} ms".format(1000 * sum(avg_max_latency)/len(self.inference_times)))
        output_function("Average Latency per Batch: {:.3f} ms".format(1000 * sum(avg_latency)/len(self.inference_times)))
        output_function("Average Inference Throughput: {:.3f} img/s".format(sum(self.inference_throughputs)/len(self.inference_throughputs)))
        output_function("--------------------------------------------------")

        if self.cfg.eval.metrics:
            self.compute_and_print_eval_metrics(output_function)


class AutoAnchors:
    """
    Class to calculate the best set of anchors for a dataset with a given image size
    """
    def __init__(self, dataset, cfg, gen):
        self.dataset = dataset
        self.image_size = cfg.image_size
        self.anchor_threshold = cfg.anchor_threshold
        self.anchors = torch.stack((torch.stack((torch.tensor(cfg.anchors.p3width, requires_grad=False), torch.tensor(cfg.anchors.p3height, requires_grad=False)), dim=1),
                                    torch.stack((torch.tensor(cfg.anchors.p4width, requires_grad=False), torch.tensor(cfg.anchors.p4height, requires_grad=False)), dim=1),
                                    torch.stack((torch.tensor(cfg.anchors.p5width, requires_grad=False), torch.tensor(cfg.anchors.p5height, requires_grad=False)), dim=1))).view(3 * 4, 2)
        self.n_anchors = self.anchors.shape[0]
        self.n_generations = gen

    def best_ratio_metric(self, k_points: Union[torch.Tensor, np.array], width_height: torch.Tensor):
        """
        Computes the best match and ratio of k points with a given width and height
        Parameters:
            k_points (Union[torch.Tensor, np.array]): k points, in this case anchors
            width_height (torch.Tensor): a width and height to compare to
        Returns:
            best_ratio_in_dim (torch.Tensor): best ratio for each anchor per label
            best (torch.Tensor): best anchor ratio for each label
        """
        k_points = torch.tensor(k_points).float()
        ratio = width_height[:, None] / k_points[None]
        best_ratio_in_dim = torch.min(ratio, 1. / ratio).min(2)[0]
        best = best_ratio_in_dim.max(1)[0]
        return best_ratio_in_dim, best

    def metric(self, k_points: np.array, width_height: torch.Tensor):  # compute metric
        """
        Computes the best possible recall and the anchors above the specified threshold
        Parameters:
            k_points (np.array): anchor box sizes
            width_height (torch.Tensor): width height of all labels
        Returns:
            best possible recall
            anchors above threshold
        """

        best_ratio_in_dim, best = self.best_ratio_metric(k_points, width_height)
        anchors_above_threshold = (best_ratio_in_dim > 1. / self.anchor_threshold).float().sum(1).mean()
        best_possible_recall = (best > (1. / self.anchor_threshold)).float().mean()
        return best_possible_recall, anchors_above_threshold

    def fitness(self, k_points: np.array, width_height: torch.Tensor):  # mutation fitness
        """
        Computes the fitness of k points with width and height
        Parameters:
            k_points (np.array): k points, in this case anchors
            width_height (torch.Tensor): a width and height to compare to
        Returns:
            fitness
        """
        _, best = self.best_ratio_metric(k_points, width_height)
        return (best * (best > (1. / self.anchor_threshold)).float()).mean()

    def mutate(self, population: np.array, mutation_prob: float = 0.9, sigma: float = 0.1):
        """
        Computes a new population scaling the original population randomly
        Parameters:
            population (np.array): original population
            mutation_prob (float): probability of mutation of the population
            sigma (float): the step of the change in the mutation
        Returns:
            a new mutated population
        """
        while (population == 1).all():
            population = ((np.random.random(population.shape) < mutation_prob) * np.random.random() * np.random.randn(*population.shape) * sigma + 1).clip(0.3, 3.0)
        return population

    def kmean_anchors(self, labels_wh: np.array):
        """
        Computes a new set of anchors using k-means and evolving mutation to find the best fit
        Parameters:
            labels_wh (np.array): labels width and height
        Returns:
            best population (set of anchors)
        """
        # Small object detection
        n_small_objects = (labels_wh < 3.0).any(1).sum()
        if n_small_objects > 0:
            print("WARNING: Small objects found in the dataset")
            print(str(n_small_objects) + " label boxes are < 3 pixels in width or height out of " + str(len(labels_wh)))
        labels_wh_filtered = labels_wh[(labels_wh >= 2.0).any(1)]  # filter > 2 pixels

        print("Running kmeans for %g anchors on %g points" % (self.n_anchors, len(labels_wh_filtered)))

        # Kmeans calculation
        sigma = labels_wh_filtered.std(0)
        k_points, dist = kmeans(labels_wh_filtered / sigma, self.n_anchors, iter=30)
        k_points *= sigma
        labels_wh_filtered = torch.tensor(labels_wh_filtered, dtype=torch.float32)
        k_points = k_points[np.argsort(k_points.prod(1))]

        # Evolve
        fitness_value = self.fitness(k_points, labels_wh_filtered)
        population_size = k_points.shape

        pbar = tqdm(range(self.n_generations), desc='Evlolving anchors with Genetic Algorithm')
        for _ in pbar:
            new_population = self.mutate(np.ones(population_size))
            possible_new_kpoints = (k_points.copy() * new_population).clip(min=2.0)
            new_fitness_value = self.fitness(possible_new_kpoints, labels_wh_filtered)
            if new_fitness_value > fitness_value:
                fitness_value, k_points = new_fitness_value, possible_new_kpoints.copy()
                pbar.desc = "Evolving anchors with Genetic Algorithm, fitness: {:.3f}".format(fitness_value)

        return k_points[np.argsort(k_points.prod(1))]

    def __call__(self) -> List[AnchorBoxes]:
        """
        Computes the best set of anchors for the dataset
        Returns:
            best anchors
        """
        images_shapes = np.asarray(self.dataset.images_shapes)
        shapes = self.image_size * images_shapes / images_shapes.max(1, keepdims=True)
        center, shift = 1.0, 0.1
        scales = np.random.uniform(center-shift, center+shift, size=(shapes.shape[0], 1))
        wh_array = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scales, self.dataset.labels)])
        wh_tensor = torch.tensor(wh_array).float()
        wh = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, self.dataset.labels)])

        best_possible_recall, anchors_above_threshold = self.metric(self.anchors, wh_tensor)
        print("Anchors/Target: {:.3f}, Best Possible Recall: {:.3f}".format(anchors_above_threshold, best_possible_recall))

        if best_possible_recall < 0.98:
            print("Attempting to generate improved anchors.")

            new_anchors = self.kmean_anchors(wh)
            new_best_possible_recall, _ = self.metric(new_anchors.reshape(-1, 2), wh_tensor)

            if new_best_possible_recall > best_possible_recall:
                new_anchors = torch.from_numpy(new_anchors).int().view(3, -1, 2)
                new_anchors.requires_grad = False
                new_anchors = [AnchorBoxes(widths=new_anchors[0, :, 0], heights=new_anchors[0, :, 1]),
                               AnchorBoxes(widths=new_anchors[1, :, 0], heights=new_anchors[1, :, 1]),
                               AnchorBoxes(widths=new_anchors[2, :, 0], heights=new_anchors[2, :, 1])]
            else:
                print('Kmean was not able to find better anchors than the originals')
                new_anchors = [AnchorBoxes(widths=self.anchors[0:4, 0], heights=self.anchors[0:4, 1]),
                               AnchorBoxes(widths=self.anchors[4:8, 0], heights=self.anchors[4:8, 1]),
                               AnchorBoxes(widths=self.anchors[8:12, 0], heights=self.anchors[8:12, 1])]
        else:
            print("Current anchors are good enough. Using the original anchors.")
            new_anchors = [AnchorBoxes(widths=self.anchors[0:4, 0], heights=self.anchors[0:4, 1]),
                           AnchorBoxes(widths=self.anchors[4:8, 0], heights=self.anchors[4:8, 1]),
                           AnchorBoxes(widths=self.anchors[8:12, 0], heights=self.anchors[8:12, 1])]

        return new_anchors


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
         (M x N) IoA between boxes1 and boxes2
    """

    inter = intersection(boxes1, boxes2)
    area2 = area(boxes2)

    return inter / area2


def iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Return intersection-over-union of boxes.
    Both sets of boxes are expected to be in (xmin, ymin, xmax, ymax) format
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


def nms(scores: torch.Tensor, boxes: torch.Tensor, classes: torch.Tensor, iou_threshold: float, max_detections: int) -> List[torch.Tensor]:
    """
    Perform non maximum suppression on predictions
    Parameters:
        scores (torch.Tensor): objectness scores per box
        boxes (torch.Tensor): (xmin, ymin, xmax, ymax)
        classes (torch.Tensor): classes per box
        iou_threshold (float):  Predictions that overlap by more than this threshold will be discarded
        max_detections (int) : Maximum number of detections per image
    Returns:
        List[torch.Tensor]: Predictions filtered after NMS, indexes, scores, boxes, classes, and the number of detection per image
    """
    batch = scores.shape[0]
    selected_box_indx = torch.full((batch, max_detections), -1, dtype=torch.long)
    cpu_classes = torch.full((batch, max_detections), torch.iinfo(torch.int32).max, dtype=int)
    cpu_boxes = torch.zeros((batch, max_detections, 4))
    cpu_scores = torch.zeros((batch, max_detections))
    cpu_true_max_detections = torch.full((batch,), max_detections)

    for i, (bscores, bboxes, bclasses) in enumerate(zip(scores, boxes, classes)):
        nms_preds = torchvision_nms(bboxes, bscores, iou_threshold)

        if nms_preds.shape[0] > max_detections:
            selected_box_indx[i] = nms_preds[:max_detections]
        else:
            selected_box_indx[i, :nms_preds.shape[0]] = nms_preds
            cpu_true_max_detections[i] = nms_preds.shape[0]

        batch_indices = selected_box_indx[i, :cpu_true_max_detections[i]]

        cpu_classes[i, :cpu_true_max_detections[i]] = bclasses[batch_indices]
        cpu_boxes[i, :cpu_true_max_detections[i]] = bboxes[batch_indices]
        cpu_scores[i, :cpu_true_max_detections[i]] = bscores[batch_indices]

    return [selected_box_indx, cpu_scores, cpu_boxes, cpu_classes.int(), cpu_true_max_detections.int()]


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
    pruned_preds_batch = post_process_prediction(y, orig_img_size, cfg)
    processed_labels_batch = post_process_labels(transformed_labels, orig_img_size, cfg)

    return pruned_preds_batch, processed_labels_batch


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
    predictions, max_n_predictions = y
    scaled_preds = []
    for i, (pred, max_n) in enumerate(zip(predictions, max_n_predictions)):
        if pred is None:
            continue
        max_n = -1 if max_n == 0 else max_n
        pred = pred[:max_n]
        pred = xyxy_to_xywh(pred)
        scaled_pred = scale_boxes_to_orig(pred, orig_img_sizes[i], cfg.model.image_size)
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
    processed_labels = []
    for i, label in enumerate(labels):
        # Remove label padding
        label = label[torch.abs(label.sum(axis=1)) != 0.]
        label = standardize_labels(label, cfg.model.image_size, cfg.model.image_size)
        scaled_boxes = scale_boxes_to_orig(label[:, 1:], orig_img_sizes[i], cfg.model.image_size)
        label[:, 1:] = scaled_boxes
        processed_labels.append(label)
    return processed_labels


def standardize_labels(labels: np.array, width: int, height: int) -> np.array:
    """
    Convert normalized label (as the ratio of image size) to the pixel scale of the image size
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


def normalize_labels(labels: np.array, width: int, height: int) -> np.array:
    """
    Convert standardized label (as the absolute pixels) to normalized scale (as a ratio) of the image size
    Parameters:
        labels (np.array): NX5 array of labels
        width (int): image width
        height (int): image height
    Returns:
        (np.array): an NX5 array of normalized labels
    """
    labels[:, [1, 3]] /= width
    labels[:, [2, 4]] /= height
    return labels


def xyxy_to_xywh(boxes: Union[np.array, torch.Tensor]) -> Union[np.array, torch.Tensor]:
    """
    Convert xmin, ymin, xmax, ymax to centerx, centery, width, height
    Parameters:
        boxes (np.array or torch.Tensor): boxes in the xmin, ymin, xmax, ymax format
    Returns:
        np.array or torch.Tensor: boxes in the centerx, centery, width, height format
    """
    width = boxes[..., 2] - boxes[..., 0]
    height = boxes[..., 3] - boxes[..., 1]

    x = boxes[..., 0] + width / 2
    y = boxes[..., 1] + height / 2

    if type(boxes).__module__ == np.__name__:
        new_boxes = np.stack((x, y, width, height), axis=-1)
        if boxes.shape[-1] != 4:
            new_boxes = np.concatenate((new_boxes, boxes[..., 4:]), axis=-1)
        return new_boxes
    else:
        new_boxes = torch.stack((x, y, width, height), axis=-1)
        if boxes.shape[-1] != 4:
            new_boxes = torch.cat((new_boxes, boxes[..., 4:]), axis=-1)
        return new_boxes


def xywh_to_xyxy(boxes: Union[np.array, torch.Tensor]) -> Union[np.array, torch.Tensor]:
    """
    Convert centerx, centery, width, height to xmin, ymin, xmax, ymax
    Parameters:
        boxes (torch.Tensor or np.array): boxes in the centerx, centery, width, height format
    Returns:
        torch.Tensor or np.array: boxes in the xmin, ymin, xmax, ymax format
    """
    xmin = boxes[..., 0] - boxes[..., 2] / 2
    ymin = boxes[..., 1] - boxes[..., 3] / 2

    xmax = xmin + boxes[..., 2]
    ymax = ymin + boxes[..., 3]

    if type(boxes).__module__ == np.__name__:
        new_boxes = np.stack((xmin, ymin, xmax, ymax), axis=-1)
        if boxes.shape[-1] != 4:
            new_boxes = np.concatenate((new_boxes, boxes[..., 4:]), axis=-1)
        return new_boxes
    else:
        new_boxes = torch.stack((xmin, ymin, xmax, ymax), axis=-1)
        if boxes.shape[-1] != 4:
            new_boxes = torch.cat((new_boxes, boxes[..., 4:]), axis=-1)
        return new_boxes


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
