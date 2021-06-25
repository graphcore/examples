# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

parser = argparse.ArgumentParser(
    description="""Convert detection result produce by evaluation.py to coco format
    that can be uploaded to c detection(Bounding Box) evaluation serverevalution in TensorFlow""")
parser.add_argument("--output-file", type=str, default="mAP/coco_detections_results.json",
                    help="output file that can be uploaded directly to coco detection(Bounding Box) evaluation server")
parser.add_argument("--coco-category-file", type=str, default="./data/classes/coco_categories.txt",
                    help="coco category_id to category name file")
parser.add_argument("--image-list", type=str, default="./data/dataset/coco_val2017.txt",
                    help="coco category_id to category name file")
parser.add_argument("--prediction-path", type=str, default="./mAP/predicted.txt",
                    help="coco category_id to category name file")
arguments = parser.parse_args()


category_ids = {}
with open(arguments.coco_category_file, "r") as f:
    for line in f:
        cat_id, name = line.strip().split(",")
        category_ids[name] = cat_id

target_file = open(arguments.output_file, "w")

label_list = []
with open(arguments.prediction_path) as f:
    previous_line = None
    for line in f:
        line = line.strip()
        if ":" in line:
            label_list.append([])
            assert int(line[:-1]) == len(label_list)-1
            continue
        label_list[-1].append(line)

with open(arguments.image_list, "r") as annotation_file:
    # load_all images
    lines = []
    for line in annotation_file:
        lines.append(line)

    target_file.write("[\n")
    # assuming that prediction result name is the file index
    # which is implemented in "evaluation.py"
    # for every image
    for i, line in enumerate(lines):
        image_path = line.strip().split()[0]
        image_id = image_path.split("/")[-1].replace(".jpg", "")
        image_id = int(image_id)
        dets = label_list[i]
        # for every detection
        for j in range(len(dets)):
            det = dets[j]
            category_id, score, xmin, ymin, xmax, ymax = det.strip().split(",")
            target_file.write('{{"image_id":{}, "category_id":{}, "bbox":[{}, {}, {}, {}], "score":{}}}'.format(
                image_id, category_ids[category_id], xmin, ymin, int(xmax)-int(xmin), int(ymax)-int(ymin), score))
            if i == len(lines)-1 and j == len(dets)-1:
                target_file.write("\n")
            else:
                target_file.write(",\n")
    target_file.write("]")
target_file.close()
