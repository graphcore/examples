# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import poptorch
import torch
import torch.nn as nn
from utils.anchors import AnchorBoxes
from utils.tools import bbox_iou
from utils.custom_ops import CopyTensor


class PreprocessTargets(nn.Module):
    """
    Implementation of IPU label preprocessing for yolov4 loss
    """
    def __init__(self, cfg, tmp_anchors=None):
        super().__init__()
        self.strides = torch.tensor(cfg.strides)
        self.num_classes = cfg.n_classes
        self.image_size = cfg.image_size
        self.input_channels = cfg.input_channels
        self.train_input_sizes = [int(self.image_size/stride) for stride in self.strides]

        self.precision = (torch.float32 if cfg.precision == 'single' else torch.float16)

        if tmp_anchors is None:
            tmp_anchors = [AnchorBoxes(widths=torch.tensor(cfg.anchors.p3width, dtype=self.precision),
                                       heights=torch.tensor(cfg.anchors.p3height, dtype=self.precision)),
                           AnchorBoxes(widths=torch.tensor(cfg.anchors.p4width, dtype=self.precision),
                                       heights=torch.tensor(cfg.anchors.p4height, dtype=self.precision)),
                           AnchorBoxes(widths=torch.tensor(cfg.anchors.p5width, dtype=self.precision),
                                       heights=torch.tensor(cfg.anchors.p5height, dtype=self.precision))]

        self.auto_anchors = cfg.auto_anchors
        self.anchor_per_scale = len(tmp_anchors[0])

        self.max_n_labels = [int(cfg.max_nlabels_p3), int(cfg.max_nlabels_p4), int(cfg.max_nlabels_p5)]

        self.anchors = [(tmp_anchor.to_torch_tensor().view(2, self.anchor_per_scale).transpose(1, 0).to(self.precision).contiguous() / self.strides[i]) for i, tmp_anchor in enumerate(tmp_anchors)]

        self.anchor_threshold = cfg.anchor_threshold

    def get_indices_tbox(self, labels, idx_detection_head):
        # Input: Labels[batch_size, max_n_labels, 5] class,x,y,w,h
        # Output: anchor_ind [5, batch_size, max_n_labels, n_anchors_per_scale, 1] indices for anchors
        #         yind [5, batch_size, max_n_labels, n_anchors_per_scale, 1] indices for y
        #         xind [5, batch_size, max_n_labels, n_anchors_per_scale, 1] indices for x
        #         t_boxes [5, batch_size, max_n_labels, n_anchors_per_scale, 5] true values for the boxes
        b_size = labels.shape[0]
        n_labels = labels.shape[1]
        g = 0.5
        off = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.]]) * g

        feature_map_size = self.train_input_sizes[idx_detection_head]

        classes = labels[:, :, 0].unsqueeze(axis=-1)
        xywh = labels[:, :, 1:]
        xywh = xywh * feature_map_size
        labels = torch.cat((classes, xywh), axis=-1)

        labels = labels.unsqueeze(axis=2).repeat(1, 1, self.anchor_per_scale, 1)
        anchor_ind = torch.arange(1., float(self.anchor_per_scale)+1.).view(1, 1, self.anchor_per_scale, 1).repeat(b_size, 1, n_labels, 1).view(b_size, n_labels, self.anchor_per_scale, 1)
        labels = torch.cat((labels, anchor_ind), axis=-1)

        # We check the difference between the ratio of all the anchors with all the labels
        # If some are under a threshold we choose them as good enough
        labels_wh = labels[:, :, :, 3:5]
        ratio_anchors = labels_wh / self.anchors[idx_detection_head].unsqueeze(axis=0).to(device=labels_wh.device)
        # labels_wh
        # tensor([[[ 2.,  2.],
        #          [ 1.,  3.],
        #          [ 1.,  1.],
        #          [10.,  5.]]])
        # anchors
        # tensor([[[2., 2.],
        #         [2., 2.],
        #         [2., 2.],
        #         [2., 2.]]])
        # ratio_a = labels_wh / anchors
        # If the ratios are bigger than the anchor then ratios will be > 1.
        # if the ratios are smaller then the anchor then 1/ratios will be > 1.
        # ratio_a
        # tensor([[[1.0000, 1.0000],
        #          [0.5000, 1.5000],
        #          [0.5000, 0.5000],
        #          [5.0000, 2.5000]]])
        # 1.0/ratio_a
        # tensor([[[1.0000, 1.0000],
        #          [2.0000, 0.6667],
        #          [2.0000, 2.0000],
        #          [0.2000, 0.4000]]])
        # We calculate the worse ratio in each dimension
        worse_ratios_wh = torch.max(ratio_anchors, 1./ratio_anchors)
        # Then we get the worst ratio for each label per anchor in any of the 2 dimensions
        # and we compare to a hyper parameter if the worst case is smaller then is a good fit
        mask = torch.amax(worse_ratios_wh, 3)[0]
        mask = torch.where(mask != 0., mask, torch.tensor([torch.finfo(torch.half).max]))
        mask = torch.lt(mask, torch.tensor([self.anchor_threshold]))

        # We mask the labels and the anchor indexes, these are the best labels for our anchors
        best_targets = torch.where(mask.unsqueeze(axis=-1).repeat(1, 1, 1, labels.shape[-1]), labels, torch.tensor([0.]))

        # We get the point x and y
        best_xy = best_targets[:, :, :, 1:3]
        # Do the inverse of the point:
        best_inverse_xy = torch.where(best_xy != 0., feature_map_size - best_xy, torch.tensor([0.]))

        # We mask as true all the pixels that the decimal part is smaller than 0.5 and not located in the right up border.
        # For example, 3.2 .2 is smaller than g=0.5 and 3. is greater than 1.
        x_mask, y_mask = ((best_xy % 1. < g) * (best_xy > 1.)).permute(3, 0, 1, 2)

        # We mask as true all the pixels for the inverse x and y that the decimal part is smaller than 0.5 and not located in the inverse right up border.
        # For example, if the original x is 3.2 and the image size is 10 then the inverse of 3.2 is "10 - 3.2 = 6.8" which is False for
        # the first comparison since .8 is greater than g=0.5 but True for the second one since 6. is greater than 1.
        # However, how we are doing a "logical and" between the comparisons "False & True = False"
        inverse_x_mask, inverse_y_mask = ((best_inverse_xy % 1. < g) * (best_inverse_xy > 1.)).permute(3, 0, 1, 2)

        # Now we have 5 masks the first one full of Trues which will get all the labels and 4 extra masks with the previous mentioned criteria
        indices = torch.stack((torch.ones_like(x_mask, dtype=int).bool(), x_mask, y_mask, inverse_x_mask, inverse_y_mask))

        # Now we mask with those combinations
        targets = torch.where(indices.unsqueeze(axis=-1).repeat(1, 1, 1, 1, best_targets.shape[-1]), best_targets.unsqueeze(axis=0).repeat(indices.shape[0], 1, 1, 1, 1), torch.tensor([0.]))

        # Offsets will decrease or increase the values for the masks where we checked if the decimal part was greater or smaller than 0.5
        # For all the x, y values smaller than 0.5 we will add 0.5 for all the values grester than 0.5 we will add -0.5
        offsets = torch.where(indices.unsqueeze(axis=-1).repeat(1, 1, 1, 1, off.shape[-1]), off.view(indices.shape[0], 1, 1, 1, off.shape[-1]).repeat(1, b_size, n_labels, self.anchor_per_scale, 1), torch.tensor([0.]))

        # We flatten the tensors to index them efficiently
        targets = targets.view(-1, targets.shape[-1])
        offsets = offsets.view(-1, offsets.shape[-1])

        # We calculate the the addition of all the elements
        # to get the non-zero elements, we sum all the dimension
        # because some classes might be 0. and the same for x, y and anchor index
        cxywh_added = torch.sum(targets, dim=-1)

        # This return is used when we want to get the maximum number of labels
        # after preprocessing
        if self.auto_anchors:
            return torch.sum(cxywh_added.bool())

        # We have lots of padding in the mask so we reduce some by taking the maximum
        # number of preprocessing images, by default is the worst case scenario
        # but it can be refined by sweeping through the dataset per image size
        _, idx_filter = torch.topk(cxywh_added, self.max_n_labels[idx_detection_head])
        idx_filter = idx_filter.float().long()

        # We use the indexes to eliminate some of the padding
        targets = torch.index_select(input=targets, dim=0, index=idx_filter)
        offsets = torch.index_select(input=offsets, dim=0, index=idx_filter)

        # We get the different elements from the targets
        classes = targets[:, 0].unsqueeze(axis=-1)
        label_xy = targets[:, 1:3]
        label_wh = targets[:, 3:5]

        # We calculate the indices of x and y
        label_xy_indices = torch.where(label_xy != 0., (label_xy - offsets).long(), torch.tensor([0.], dtype=torch.long))

        # Make sure that x, y, and anchors stay within their range
        x_ind, y_ind = label_xy_indices.permute(1, 0).clamp_(0, feature_map_size-1).long()
        anchor_ind = (targets[:, 5] - 1).clamp_(0, 4).long()

        # Finally, we create the t_boxes with class, x_offset, y_offset, width and height
        t_boxes = torch.cat((classes, (label_xy - label_xy_indices), label_wh), axis=-1)
        return anchor_ind, y_ind, x_ind, t_boxes

    def forward(self, real_labels):
        t_indices_boxes_p3 = self.get_indices_tbox(real_labels, idx_detection_head = 0)
        t_indices_boxes_p4 = self.get_indices_tbox(real_labels, idx_detection_head = 1)
        t_indices_boxes_p5 = self.get_indices_tbox(real_labels, idx_detection_head = 2)

        return t_indices_boxes_p3, t_indices_boxes_p4, t_indices_boxes_p5


class Yolov4_loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cp, self.cn = self.smooth_BCE(eps=0.0)

        self.preprocess_targets = PreprocessTargets(cfg.model)
        self.image_size = self.preprocess_targets.image_size

        self.ciou_ratio = cfg.training.ciou_ratio
        self.anchors = self.preprocess_targets.anchors
        self.prediction_heads = len(self.anchors)
        self.n_classes = cfg.model.n_classes

        self.cpu_mode = not cfg.model.ipu
        self.copy_tensor = CopyTensor(self.cpu_mode)

        # balance constants for 3 prediction heads (P3-P5), 4 (P3-P6) or 5 (P3-P7)
        balance_switch = {
            3: [4.0, 1.0, 0.4],
            4: [4.0, 1.0, 0.4, 0.1],
            5: [4.0, 1.0, 0.5, 0.4, 0.1]
        }
        self.balance = balance_switch[self.prediction_heads]
        self.object_scaling = (1.4 if self.prediction_heads >= 4 else 1.)

        self.precision = self.preprocess_targets.precision

        # TODO implement FocalLoss when fl_gamma > 0.0 https://phabricator.sourcevertex.net/T55876
        self.fl_gamma = cfg.training.fl_gamma
        self.box_gain = cfg.training.box_gain
        self.class_gain = cfg.training.class_gain
        self.object_gain = cfg.training.object_gain

    def compute_box_loss(self, flattened_prediction, indices, mask, tbox, head_idx, anchor_ind):

        if mask.sum() > 0:
            filtered_prediction = torch.index_select(input=flattened_prediction, dim=0, index=indices)
            p = torch.sigmoid(filtered_prediction[:, :4]) * torch.tensor([2.], dtype=torch.float32)
            pxy = p[:, :2] - torch.tensor([0.5], dtype=torch.float32)
            pwh = p[:, 2:4]
            anchor_sizes = self.anchors[head_idx][anchor_ind.long()].float()
            pwh = pwh * pwh * anchor_sizes
            pbox = torch.cat((pxy, pwh), -1)
            ciou = bbox_iou(pbox, tbox, is_xyxy=False, special_iou_type='ciou')

            ciou_sum = torch.zeros([2])
            ciou_sum = ciou_sum.scatter_add(0, index=mask.long(), src=ciou)
            div = mask.sum().float() + torch.finfo(torch.float32).eps
            mean = ciou_sum[1]/div
            return (torch.tensor([1.]) - mean)
        else:
            return torch.tensor([0.], dtype=torch.float32)

    def compute_class_loss(self, flattened_prediction, indices, mask, classes):

        if mask.sum() > 0:
            filtered_prediction = torch.index_select(input=flattened_prediction, dim=0, index=indices)
            rows = torch.arange(classes.shape[0]).long()

            # concat prediction with an extra dimension to throw padded class in
            pred_classes = torch.cat([filtered_prediction[:, 5:].float(), torch.zeros([classes.shape[0], 1])], dim=1)
            true_classes = torch.ones(pred_classes.shape, dtype=torch.float32) * self.cn
            classes = classes.long() - 1  # shift class back after we already filter out the padded class

            # assign value to each label, all padding will go to the temp dimension
            true_classes[rows, classes] = self.cp  # switch to use index_put() for better efficiency once available

            bce_cls = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_classes.float(), true_classes.float(), reduction='none', pos_weight=torch.Tensor([1.])
            )
            bce_cls = bce_cls[:, :self.n_classes]  # remove the temp dimension
            bce_sum = torch.zeros([2, bce_cls.shape[1]])
            bce_sum = bce_sum.scatter_add(0, index=mask.view(-1, 1).repeat(1, self.n_classes).long(), src=bce_cls)
            div = (mask.sum() * bce_cls.shape[1]).float() + torch.finfo(torch.float32).eps
            return bce_sum[1].sum()/div
        else:
            return torch.tensor([0.], dtype=torch.float32)

    def compute_object_loss(self, flattened_prediction, indices, mask, tbox, head_idx, anchor_ind):

        prediction_objectness = self.copy_tensor(flattened_prediction[:, 4])[0]
        true_objectness = torch.zeros((prediction_objectness.shape[0]) + 1)

        if mask.sum() > 0:
            filtered_prediction = torch.index_select(input=flattened_prediction, dim=0, index=indices)
            p = torch.sigmoid(filtered_prediction[:, :4]) * torch.tensor([2.], dtype=torch.float32)
            pxy = p[:, :2] - torch.tensor([0.5], dtype=torch.float32)
            pwh = p[:, 2:4]
            anchor_sizes = self.anchors[head_idx][anchor_ind.long()].float()
            pwh = pwh * pwh * anchor_sizes
            pbox = torch.cat((pxy, pwh), -1)
            ciou = bbox_iou(pbox, tbox, is_xyxy=False, special_iou_type='ciou')

            objectness = (torch.tensor([1.]) - self.ciou_ratio) + self.ciou_ratio * ciou.detach().clamp(0).float()
            mod_indices = torch.where(mask == 0, flattened_prediction.shape[0], indices)
            true_objectness[mod_indices] = objectness

        return torch.nn.functional.binary_cross_entropy_with_logits(
            prediction_objectness.float(), true_objectness[:flattened_prediction.shape[0]], pos_weight=torch.Tensor([1.])
        ) * self.balance[head_idx]

    def forward(self, predictions, targets):

        processed_targets = self.preprocess_targets(targets)
        box_loss, object_loss, class_loss = torch.zeros(1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)

        for head_idx, prediction in enumerate(predictions):
            # preprocess targets and mask for each prediction head
            anchor_ind, y_ind, x_ind, t_boxes = processed_targets[head_idx]
            mask = torch.where(t_boxes[:, 0] == torch.tensor([0.], dtype=torch.float32), torch.tensor([0.], dtype=torch.float32), torch.tensor([1.], dtype=torch.float32))
            prediction_size = torch.tensor(prediction.shape, dtype=int)
            flattened_prediction = prediction.view(-1, prediction_size[-1]).float()
            indices = self.flatten_indices(prediction_size, anchor_ind, y_ind, x_ind)
            t_boxes = t_boxes.float()
            # compute loss
            box_loss = box_loss + self.compute_box_loss(flattened_prediction, indices, mask, t_boxes[:, 1:], head_idx, anchor_ind)
            object_loss = object_loss + self.compute_object_loss(flattened_prediction, indices, mask, t_boxes[:, 1:], head_idx, anchor_ind)
            class_loss = class_loss + self.compute_class_loss(flattened_prediction, indices, mask, t_boxes[:, 0])

        box_loss = box_loss * self.box_gain
        object_loss = object_loss * self.object_gain * self.object_scaling
        class_loss = class_loss * self.class_gain
        loss = box_loss + object_loss + class_loss
        return poptorch.identity_loss(loss, reduction='none'), box_loss.detach(), object_loss.detach(), class_loss.detach()

    def smooth_BCE(self, eps=0.1):
        # return positive, negative label smoothing BCE targets
        return 1.0 - 0.5 * eps, 0.5 * eps

    def flatten_indices(self, prediction_size, anchor_ind, y_ind, x_ind):
        # transform index from [1, anchor_per_scale, y_size, x_size, num_classes + 5] format to [anchor_per_scale * y_size * x_size, num_classes + 5]
        anchor_ind = anchor_ind.int() * (torch.prod(prediction_size[2:4]))
        y_ind = y_ind.int() * (prediction_size[3])
        x_ind = x_ind.int()
        indices = (anchor_ind + y_ind + x_ind).long()
        return indices
