# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import collections
from datetime import datetime
import json
import math
import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ruamel import yaml
from scipy.cluster.vq import kmeans
import torch
import time
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
import wandb
from yacs.config import CfgNode

from utils.anchors import AnchorBoxes


class StatRecorder:
    """
    Records and prints the time stats and metrics of a model (latency and throughput)
    """
    def __init__(self, cfg: CfgNode, data_path: str, log_wandb: bool = False):
        """
        Collecting latency and evaluation stats for inference and post processing
        Parameters:
            cfg: yacs object containing the config
            class_names: list of all the class names
        """
        # Inference stat initialization
        self.total_times = []
        self.inference_times = []
        self.inference_throughputs = []
        self.total_throughputs = []
        self.eval_stats = []
        self.bbox_summary = []
        self.image_count = cfg.model.micro_batch_size * cfg.ipuopts.device_iterations
        self.cfg = cfg
        coco_metadata = yaml.safe_load(open(os.environ['PYTORCH_APPS_DETECTION_PATH'] + "/" + cfg.model.class_name_path))
        self.class_names = coco_metadata['class_names']
        self.coco_91_class = coco_metadata['coco_91_class']
        self.seen = 0
        self.iou_values = torch.linspace(0.5, 0.95, 10)
        self.num_ious = self.iou_values.numel()
        self.data_path = data_path

        # Training stat initialization
        self.log_wandb = log_wandb
        self.loss_keys = ['mean_box', 'mean_obj', 'mean_cls', 'mean_total']
        self.loss_items = torch.zeros(4)
        self.reset_train_stats()
        self.epoch = 0

        if self.log_wandb:
            wandb.init(project="pytorch-detection", config=dict(cfg))
            if cfg.model.mode == 'train':
                wandb.define_metric("epoch")
                wandb.define_metric("epoch_total_loss", step_metric="epoch")
                wandb.define_metric("epoch_total_obj_loss", step_metric="epoch")
                wandb.define_metric("epoch_total_cls_loss", step_metric="epoch")
                wandb.define_metric("epoch_total_box_loss", step_metric="epoch")
                wandb.define_metric("precision", step_metric="epoch")
                wandb.define_metric("recall", step_metric="epoch")
                wandb.define_metric("m_ap", step_metric="epoch")
                wandb.define_metric("m_ap50", step_metric="epoch")
                wandb.save(os.environ['PYTORCH_APPS_DETECTION_PATH'] + "/models/loss.py")

    def reset_train_stats(self):
        self.sum_all_loss, self.sum_box_loss, self.sum_obj_loss, self.sum_cls_loss = 0.0, 0.0, 0.0, 0.0
        self.moving_avg_loss = torch.zeros(4)
        self.throughput = 0.0

    def reset_eval_stats(self):
        self.eval_stats = []
        self.seen = 0
        self.bbox_summary = []

    def record_eval_stats(self, labels: np.array, predictions: np.array, image_size: torch.Tensor, image_id: str, run_coco_eval: bool):
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
            bboxes = xywh_to_xyxy(predictions[:, :4]).detach()
            scores = predictions[:, 4].cpu().detach().numpy()
            class_pred = predictions[:, 5].cpu().detach().numpy()

            clip_coords(bboxes, image_size)

            # save the predictions in json format for COCOeval [{"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}]
            if run_coco_eval:
                for i, bbox in enumerate(bboxes):
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    self.bbox_summary.append({
                        "image_id": int(image_id),
                        "category_id": self.coco_91_class[int(class_pred[i])],
                        "bbox": [bbox[0].item(), bbox[1].item(), width.item(), height.item()],
                        "score": scores[i].item()
                    })
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

            self.eval_stats.append((correct.cpu().detach().numpy(), scores, class_pred, target_cls))

    def compute_and_print_eval_metrics(self, output_function):
        """
        Computes and prints the evaluation metrics
        """
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        precision, recall, f1, mean_precision, mean_recall, m_ap50, m_ap = 0., 0., 0., 0., 0., 0., 0.
        ap = []
        eval_stats = [np.concatenate(x, 0) for x in zip(*self.eval_stats)]
        valid_eval = len(eval_stats) and eval_stats[0].any()
        if valid_eval:
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
        if self.cfg.eval.verbose and valid_eval:
            for indx, cls in enumerate(ap_class):
                output_function(pf % (self.class_names[cls], self.seen, nt[cls], precision[indx], recall[indx], ap50[indx], ap[indx]))

        return self.seen, nt.sum(), mean_precision, mean_recall, m_ap50, m_ap

    def write_and_eval_coco(self):
        temp_pred_file = "temp_detections_{}.json".format(datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-4])
        with open(temp_pred_file, "w") as f:
            json.dump(self.bbox_summary, f)
        annotation_file = self.data_path + '/' + self.cfg.dataset.name + '/annotations/' + self.cfg.dataset.test.annotation
        try:
            ground_truth = COCO(annotation_file)
            predictions = ground_truth.loadRes(temp_pred_file)
            coco_eval = COCOeval(ground_truth, predictions, iouType='bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        except Exception as e:
            print("pycocotools failed with: ", e)
        os.remove(temp_pred_file)

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

    def record_training_stats(self, box_loss, object_loss, class_loss, total_loss, step_idx, num_img_per_step, throughput=None):
        box_loss = box_loss.mean().view(1)
        object_loss = object_loss.mean().view(1)
        class_loss = class_loss.mean().view(1)
        total_loss = total_loss.mean().view(1)
        self.sum_all_loss += total_loss.item()
        self.sum_box_loss += box_loss.item()
        self.sum_obj_loss += object_loss.item()
        self.sum_cls_loss += class_loss.item()
        if throughput is not None:
            self.throughput += throughput
            self.avg_throughput = self.throughput / (step_idx + 1)

        self.loss_items = torch.cat((box_loss, object_loss, class_loss, total_loss)) / num_img_per_step  # normalize per image
        self.moving_avg_loss = (self.moving_avg_loss * step_idx + self.loss_items) / (step_idx + 1)

    def log_train_step(self, optimizer_state, step_idx, output_func):
        if self.log_wandb:
            log_dict = {
                "total_loss": self.loss_items[3],
                "box_loss": self.loss_items[0],
                "obj_loss": self.loss_items[1],
                "cls_loss": self.loss_items[2],
                "pg0_lr": optimizer_state[0]['lr'],
                "pg1_lr": optimizer_state[1]['lr'],
                "pg2_lr": optimizer_state[2]['lr'],
                "pg0_momentum": optimizer_state[0]['momentum'],
                "pg1_momentum": optimizer_state[1]['momentum'],
                "pg2_momentum": optimizer_state[2]['momentum'],
                "pg1_weight_decay": optimizer_state[1]['weight_decay'],
                "throughput": self.avg_throughput,
            }
            mean_losses = {self.loss_keys[i]: self.moving_avg_loss.tolist()[i] for i in range(len(self.loss_keys))}
            wandb.log({**log_dict, **mean_losses})
        output_func("step: ", step_idx, " total loss: ", self.loss_items[3].item(),
                    " box loss: ", self.loss_items[0].item(),
                    " obj loss: ", self.loss_items[1].item(),
                    " cls loss: ", self.loss_items[2].item(),
                    "moving avg loss: ", self.moving_avg_loss,
                    " throughput: ", self.avg_throughput, "samples/sec")

    def log_train_epoch(self, output_func):
        if self.log_wandb:
            log_epoch = {
                "epoch_total_loss": self.sum_all_loss,
                "epoch_total_box_loss": self.sum_box_loss,
                "epoch_total_obj_loss": self.sum_obj_loss,
                "epoch_total_cls_loss": self.sum_cls_loss,
                "epoch": self.epoch
            }
            wandb.log(log_epoch)
        output_func("Epoch ", self.epoch, ": total loss: ", self.sum_all_loss, " box loss: ", self.sum_box_loss, " obj loss: ", self.sum_obj_loss, " class loss: ", self.sum_cls_loss)
        self.epoch = self.epoch + 1

    def logging(self, output_function, run_coco_eval):
        """Prints using the "output_function" given the average of the times recorded during the call to record_inference_stats()
            Parameters:
                output_function (function): function used to print the stats recorded
        """
        avg_min_latency = [x[0] for x in self.inference_times]
        avg_max_latency = [x[1] for x in self.inference_times]
        avg_latency = [x[2] for x in self.inference_times]

        output_function("Inference stats: image size {}x{}, device iterations {}, batch size {}, {} steps".format(
            self.cfg.model.image_size, self.cfg.model.image_size, self.cfg.ipuopts.device_iterations, self.cfg.model.micro_batch_size, len(self.total_times)
        ))
        output_function("--------------------------------------------------")
        output_function("Inference")
        output_function("Average Min Latency per Batch: {:.3f} ms".format(1000 * sum(avg_min_latency)/len(self.inference_times)))
        output_function("Average Max Latency per Batch: {:.3f} ms".format(1000 * sum(avg_max_latency)/len(self.inference_times)))
        output_function("Average Latency per Batch: {:.3f} ms".format(1000 * sum(avg_latency)/len(self.inference_times)))
        output_function("throughput: {:.3f} samples/sec".format(sum(self.inference_throughputs)/len(self.inference_throughputs)))
        output_function("--------------------------------------------------")

        if self.cfg.eval.metrics:
            seen, num_target, precision, recall, m_ap50, m_ap = self.compute_and_print_eval_metrics(output_function)
            if self.log_wandb:
                eval_stats = {
                    "precision": precision,
                    "recall": recall,
                    "m_ap": m_ap,
                    "m_ap50": m_ap50,
                    "epoch": self.epoch
                }
                wandb.log(eval_stats)
                wandb.run.summary["seen"] = seen
                wandb.run.summary["num_target"] = num_target

            if run_coco_eval:
                self.write_and_eval_coco()


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
    model = torch.load(weight_path, map_location=torch.device('cpu'))
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


def load_and_fuse_pretrained_weights(model: torch.nn.Module, weight_path: str, is_inference: bool = True) -> torch.nn.Module:
    """
    Given a model, load pretrained weight and fuse conv layers
    Parameters:
        model (torch.nn.Module): Placeholder model to load weights into
        weight_path: Path to the model weights
        is_inference: Fuse conv and batch norm layers when loading the model for inference
    Returns:
        A model fused with the weights
    """
    if is_inference:
        model.optimize_for_inference(fuse_all_layers=False)
    fused_state_dict = load_weights(weight_path)
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
    union = area1[:, None] + torch.finfo(torch.float32).eps + area2 - inter
    return inter / union


def bbox_iou(predicted_boxes: torch.Tensor, target_boxes: torch.Tensor, is_xyxy: bool, special_iou_type: str ='ciou'):
    """
    Calculate distance IoU (DIoU) or complete IoU (CIoU) between N pairs of predicted and target boxes,
    used specifically in the loss function.
        Arguments:
        predicted_boxes (torch.Tensor): a NX4 tensor of boxes
        target_boxes (torch.Tensor): a NX4 tensor of boxes
        is_xyxy (bool): whether the format of the box is in xyxy or xywh
        special_iou_type (str): options between ciou or diou. Default is ciou. If None is given, the iou will be returned.
    Returns:
        torch.Tensor: the 1xN matrix containing the pairwise special IoU values for each predicted-target box.
    """
    if not is_xyxy:
        predicted_boxes = xywh_to_xyxy(predicted_boxes)
        target_boxes = xywh_to_xyxy(target_boxes)

    b1x1, b1y1, b1x2, b1y2 = predicted_boxes[:, 0], predicted_boxes[:, 1], predicted_boxes[:, 2], predicted_boxes[:, 3]
    b2x1, b2y1, b2x2, b2y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
    width1, height1 = b1x2 - b1x1, b1y2 - b1y1
    width2, height2 = b2x2 - b2x1, b2y2 - b2y1

    inter_x = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0)
    inter_y = (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    inter = inter_x * inter_y

    union = (width1 * height1 + torch.finfo(torch.float32).eps) + width2 * height2 - inter
    iou_value = (inter / union)

    if not special_iou_type:
        return iou_value

    convex_width = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    convex_height = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    convex_diag_squared = convex_width * convex_width + convex_height * convex_height + torch.finfo(torch.float32).eps
    rho_squared = ((b2x1 + b2x2) - (b1x1 + b1x2)) ** 2 / 4 + ((b2y1 + b2y2) - (b1y1 + b1y2)) ** 2 / 4

    if special_iou_type == 'diou':
        return iou_value - rho_squared / convex_diag_squared
    elif special_iou_type == 'ciou':
        pi = torch.tensor([math.pi], dtype=torch.float32)
        atan1 = torch.atan(width1 / (height1 + torch.finfo(torch.float32).eps))
        atan2 = torch.atan(width2 / (height2 + torch.finfo(torch.float32).eps))
        atan_diff = atan2 - atan1
        v = (4 / (pi * pi)) * (atan_diff * atan_diff)
        with torch.no_grad():
            alpha = v / (1 - iou_value + v + torch.finfo(torch.float32).eps)
        return iou_value - (rho_squared / convex_diag_squared + v * alpha)
    else:
        raise ValueError("Type {} not supported. Valid options are 'ciou', 'diou' or None".format(special_iou_type))


def isclose(tensor1, tensor2):
    return torch.abs(tensor1-tensor2) < 1e-6


def sparse_mean(x: torch.Tensor, excluded_value=torch.zeros(1).float()):
    """
    Find the mean among the non-excluded value in the given tensor
    Parameters:
        x (torch.Tensor): input tensor for the mean to be calculated
        excluded_value (torch.Tensor): a single value to be excluded from the calculation in torch.Tensor
    Returns:
        (torch.Tensor): a mean value in torch.Tensor form
    """
    const_zero = torch.zeros(1).float()
    mask = ~isclose(x, excluded_value)
    div = (mask).sum().float()
    div = div + ((div == const_zero).float())  # removing this cause nan
    return torch.where(isclose(div, const_zero), const_zero, x.sum() / div)


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
