# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import ast
import os

from tqdm import tqdm


if __name__ == "__main__":
    filenames = []
    labels = []
    map_file = "data/imagenet1k/validation/val_official.json"
    images_path = os.path.dirname(map_file)

    with open(map_file, "r", encoding="utf-8") as f:

        all_lines = f.read().splitlines()
        print("read the json file ...")
        for line in tqdm(all_lines):
            line_d = ast.literal_eval(line)
            filename = line_d["filename"]
            label = line_d["label"]
            if os.path.exists(os.path.join(images_path, filename)):
                filenames.append(filename)
                labels.append(label)

    clean_file = os.path.join(images_path, "val_official_clean.csv")
    with open(clean_file, "w", encoding="utf-8") as f:
        for i in range(len(filenames)):
            f.write(filenames[i] + "\t" + labels[i] + "\n")
