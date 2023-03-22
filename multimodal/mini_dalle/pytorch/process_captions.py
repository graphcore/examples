# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import json
import os
import argparse


def main(captions, out_dir):
    with open(captions, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = dict()
    pairs_total = 0
    texts_total = 0
    for cap in data["annotations"]:
        image_id = str(cap["image_id"])
        image_id = (12 - len(image_id)) * "0" + image_id + ".txt"
        try:
            pairs[image_id].append(cap["caption"])
        except:
            pairs[image_id] = []
            pairs[image_id].append(cap["caption"])

    print(f"Begin to save captions to {out_dir}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for k, v in pairs.items():
        with open(os.path.join(out_dir, k), "w") as f:
            for text in v:
                if "\n" in text[:-1]:
                    text = text.replace("\n", " ")
                    while "  " in text:
                        text = text.replace("  ", " ")
                    text = text.strip("'")
                    text = text.replace(". .", ".")
                if " ." == text[-2:]:
                    text = text.replace(" .", ".")
                text = text.replace(" .", " ")
                text = text.strip()
                if len(text) > 0:
                    f.write(text + "\n")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MS COCO cpations")
    parser.add_argument(
        "--captions",
        type=str,
        default="./data/COCO/annotations/captions_train2017.json",
        help="captions annotation file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./data/COCO/train2017_captions",
        help="Directory storing processed caption files",
    )

    args = parser.parse_args()
    main(args.captions, args.out_dir)
