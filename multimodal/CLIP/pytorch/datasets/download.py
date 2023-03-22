# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 mlfoundations

# This file has been modified by Graphcore

import argparse
import multiprocessing as mp
import os
import random
import string
from io import BytesIO

import numpy as np
import requests
from PIL import Image


def generate_name():
    random_str = "".join(random.sample(string.ascii_letters + string.digits, 20))

    return random_str + ".jpg"


def grab(line):
    try:
        caption, url = line.split("\t")
    except:
        print("Parse error")
        return

    # Let's not crash if anything werid happens
    try:
        dat = requests.get(url, timeout=20)
        if dat.status_code != 200:
            print("404 file", url)
            return

        # Try to parse this an Image file, we'll fail out if not
        im = Image.open(BytesIO(dat.content))
        im.thumbnail((512, 512), Image.BICUBIC)
        if min(*im.size) < max(*im.size) / 3:
            print("Too small", url)
            return

        img_name = generate_name()
        while os.path.exists(f"{image_path}/{img_name}"):
            img_name = generate_name()

        im.save(f"{image_path}/{img_name}")

        # Another try/catch just because sometimes saving and re-loading
        # the image is different than loadding it once.
        try:
            o = Image.open(f"{image_path}/{img_name}")
            o = np.array(o)

            print("Success", o.shape, url)

            return caption, img_name

        except:
            print("Failed", url)

    except Exception as e:
        print("Unknown error", e)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download cc3m dataset")
    parser.add_argument("--url_file", type=str, help="The path of input file")
    parser.add_argument("--save_path", type=str, help="The path to save the downloaded files")

    args = parser.parse_args()

    file = args.url_file
    save_path = args.save_path
    image_path = args.save_path + "/images"

    p = mp.Pool(100)

    results = p.map(grab, [line for i, line in enumerate(open(file, "r", encoding="utf-8").read().split("\n"))])

    out = open(f"{save_path}/img_cap.csv", "w")

    for row in results:
        if row is None:
            continue

        caption, img_name = row

        if os.path.exists(f"{image_path}/{img_name}"):
            out.write(f"{img_name}\t{caption}\n")
        else:
            print("Drop", id)
    out.close()

    p.close()
