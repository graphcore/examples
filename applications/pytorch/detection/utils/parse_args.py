# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse

opt_to_cfg_map = {
    "num_ipus": "system.num_ipus",
    "num_workers": "system.num_workers",
    "input_channels": "model.input_channels",
    "activation": "model.activation",
    "normalization": "model.normalization",
    "num_classes": "model.n_classes",
    "class_name_path": "model.class_name_path",
    "image_size": "model.image_size",
    "micro_batch_size": "model.micro_batch_size",
    "mode": "model.mode",
    "half": "model.half",
    "ipu": "model.ipu",
    "batches_per_step": "ipuopts.batches_per_step",
    "class_conf_threshold": "inference.class_conf_threshold",
    "obj_threshold": "inference.obj_threshold",
    "iou_threshold": "inference.iou_threshold",
    "plot_output": "inference.plot_output",
    "plot_step": "inference.plot_step",
    "plot_dir": "inference.plot_dir",
    "dataset_name": "dataset.name",
    "max_bbox_per_scale": "dataset.max_bbox_per_scale",
    "train_file": "dataset.train.file",
    "test_file": "dataset.test.file",
    "no_eval": "eval.metrics",
    "verbose": "eval.verbose",
    "pre_nms_topk_k": "inference.pre_nms_topk_k",
    "nms_max_detections": "inference.nms_max_detections",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', type=str, default='/localdata/datasets/', help='Path to the dataset root dir (default: /localdata/datasets/)')
    parser.add_argument(
        '--config', type=str, default='configs/inference-yolov4p5.yaml', help='Configuration of the model (default: configs/inference-yolov4p5.yaml)')
    parser.add_argument('--show-config', action="store_true",
                        default=False, help='Show configuration for the program (default: False)')
    parser.add_argument('--weights', type=str, help='Pretrained weight path to use if specified (default: None)')

    # The arguments below can be used to overwrite the config
    parser.add_argument('--num-ipus', type=int, help='Number of IPUs to use (default: 1)')
    parser.add_argument('--num-workers', type=int, help='Number of workers to use (default: 20)')

    parser.add_argument('--input-channels', type=int, help='Number of channels in the input image (default: 3)')
    parser.add_argument('--activation', type=str, help='Activation function to use in the model (default: mish)')
    parser.add_argument('--normalization', type=str, help='Normalization function to use in the model (default: batch)')
    parser.add_argument('--num-classes', type=int, help='Number of classes of the model (default: 80)')
    parser.add_argument('--class-name-path', type=str, help='Path to the class names yaml (default: ./configs/class_name.yaml)')
    parser.add_argument('--image-size', type=int, help='Size of the input image (default: 896)')
    parser.add_argument('--micro-batch-size', type=int, help='The number of samples calculated in one full forward/backward pass (default: 1)')
    parser.add_argument('--mode', type=str, help='Mode to run the model (default: test)')
    parser.add_argument('--full', dest='half', action="store_false", default=True, help='Full precision (default: False)')
    parser.add_argument('--benchmark', action='store_true', default=False, help='Run performance benchmark (default: False)')

    parser.add_argument('--cpu', dest='ipu', action="store_false", default=True, help='Use cpu to run model (default: True)')
    parser.add_argument('--batches-per-step', type=int, help='Number of batches per step (default: 1)')

    parser.add_argument('--pre-nms-topk-k', type=int, help='Defines how many elements will be chosen to be process in NMS')
    parser.add_argument('--nms-max-detections', type=int, help='Number of output detections after NMS.')

    parser.add_argument('--class-conf-threshold', type=float, help='Minimum threshold for class prediction probability (default: 0.4)')
    parser.add_argument('--obj-threshold', type=float, help='Minimum threshold for the objectness score (default: 0.4)')
    parser.add_argument('--iou-threshold', type=float, help='Minimum threshold for IoU used in NMS (default: 0.65)')
    parser.add_argument('--plot-output', action='store_true', default=False, help='Plot the output during inference (default: False)')
    parser.add_argument('--plot-step', type=int, help='Plot every n image (default: 250)')
    parser.add_argument('--plot-dir', type=str, help='Directory for storing the plot output (default: plots)')

    parser.add_argument('--dataset-name', type=str, help='Name of the dataset (default: coco)')
    parser.add_argument('--max-bbox-per-scale', type=int, help='Maximum number of bounding boxes per image (default: 90)')

    parser.add_argument('--train-file', type=str, help='Path to the train annotations (default: train2017.txt)')
    parser.add_argument('--test-file', type=str,  help='Path to the test annotations (default: val2017.txt)')

    parser.add_argument('--no-eval', action='store_false', default=True, help='Dont compute the precision recall metrics (default: True)')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print out class wise eval (default: False)')

    return parser.parse_args()
