# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import yacs
import torch
import poptorch
import numpy as np
from PIL import Image
from ruamel import yaml
from run import ipu_options
from typing import NamedTuple
from typing import List, Tuple
from poptorch import inferenceModel
from models.yolov4_p5 import Yolov4P5
from utils.config import get_cfg_defaults
from torchvision.transforms import Compose
from utils.visualization import plotting_tool
from utils.postprocessing import post_processing
from utils.tools import load_and_fuse_pretrained_weights
from utils.preprocessing import ResizeImage, Pad, ToNumpy, ToTensor


class YoloOutput(NamedTuple):
    x_top_left: float
    y_top_left: float
    width: float
    height: float
    detection_confidence: float
    detected_class_id: float
    detected_class_name: str


class YOLOv4InferencePipeline:
    def __init__(self, checkpoint_path: str):
        self.cfg = get_cfg_defaults()
        self.config = "configs/inference-yolov4p5.yaml"
        self.cfg.merge_from_file(self.config)
        self.cfg.freeze()
        self.checkpoint = checkpoint_path
        self.model = self.prepare_inference_yolo_v4_model_for_ipu()
        self.class_names = yaml.safe_load(open("configs/class_name.yaml"))["class_names"]

    def __call__(self, image: Image) -> Tuple[List[YoloOutput], List[np.array]]:
        transformed_image, size = self.preprocess_image(image)
        y = self.model(transformed_image)
        dummy_label = []
        processed_batch = post_processing(self.cfg, y, [size], dummy_label)
        batch_as_list = processed_batch[0][0].tolist()
        named_output = [
            YoloOutput._make(obj[0:5] + [obj[-1]] + [self.class_names[int(obj[-1])]]) for obj in batch_as_list
        ]

        return named_output

    def preprocess_image(self, img: Image) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = img.size
        print("original image dimensions:\nh:", height, "w:", width)

        img_conv = img.convert("RGB")

        # Change the data type of the dataloader depending of the options
        if self.cfg.model.uint_io:
            image_type = "uint"
        elif not self.cfg.model.ipu or not self.cfg.model.half:
            image_type = "float"
        else:
            image_type = "half"

        size = torch.as_tensor(img_conv.size)
        resize_img_mthd = ResizeImage(self.cfg.model.image_size)
        print("img_size: ", self.cfg.model.image_size)
        pad_mthd = Pad(self.cfg.model.image_size)
        image_to_tensor_mthd = Compose([ToNumpy(), ToTensor(int(self.cfg.dataset.max_bbox_per_scale), image_type)])
        dummy_label = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
        transformed_image, transformed_labels = resize_img_mthd((img_conv, dummy_label))
        transformed_image, transformed_labels = pad_mthd((transformed_image, transformed_labels))
        transformed_image, transformed_labels = image_to_tensor_mthd((transformed_image, transformed_labels))
        transformed_image, _ = torch.unsqueeze(transformed_image, dim=0), torch.unsqueeze(transformed_labels, dim=0)

        return transformed_image, size

    def prepare_inference_yolo_v4_model_for_ipu(self) -> poptorch.PoplarExecutor:
        mode = "inference"
        original_model = Yolov4P5
        model = original_model(self.cfg)

        # Insert the pipeline splits if using pipeline
        if self.cfg.model.pipeline_splits:
            named_layers = {name: layer for name, layer in model.named_modules()}
            for ipu_idx, split in enumerate(self.cfg.model.pipeline_splits):
                named_layers[split] = poptorch.BeginBlock(ipu_id=ipu_idx + 1, layer_to_call=named_layers[split])

        model = load_and_fuse_pretrained_weights(model, self.checkpoint)
        model.optimize_for_inference()
        model.eval()

        # Create the specific ipu options if self.cfg.model.ipu
        ipu_opts = ipu_options(self.cfg, model, mode) if self.cfg.model.ipu else None

        model = inferenceModel(model, ipu_opts)

        return model

    def plot_img(self, processed_batch: List[YoloOutput], original_img: Image) -> List[str]:
        list_of_predictions = [torch.Tensor([list(p)[0:6] for p in processed_batch])]
        img_path = plotting_tool(self.cfg, list_of_predictions, [original_img])
        return img_path

    def detach(self):
        if self.model.isAttachedToDevice():
            self.model.detachFromDevice()
