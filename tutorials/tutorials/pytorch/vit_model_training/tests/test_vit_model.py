# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pathlib
import sys
import pytest
import shutil
import tutorials_tests.testing_util as testing_util
from PIL import Image
import numpy as np
import os
import csv


@pytest.mark.category2
@pytest.mark.ipus(4)
def test_vit_model(data_folder):
    """
    Test running the model with a fake dataset
    Checks if a valid roc_auc metric is printed
    """
    working_directory = pathlib.Path(__file__).absolute().parent

    out = testing_util.run_command_fail_explicitly(
        [sys.executable, "../walkthrough.py", "--dataset-dir", data_folder],
        working_directory,
    )
    roc_auc = 0.0
    for line in out.split("\n"):
        if line.find("eval_roc_auc") != -1:
            roc_auc = float(line.split("=")[-1].strip()[:-1])
            break
    # Since we are feeding random data heavily biased to No Finding
    # we expect a score around
    assert (roc_auc > 0.35) and (roc_auc < 0.65)


@pytest.fixture
def data_folder(tmp_path):
    img_size = (224, 224)
    data_folder = tmp_path / "fake_nih_dataset"
    image_folder = data_folder / "images"

    image_folder.mkdir(parents=True)

    # Create fake random grayscale images
    for i in range(0, 10000):
        img = Image.fromarray(np.random.randint(255, size=img_size, dtype=np.uint8))
        img.save(image_folder / (str(i) + ".png"))

    # Create fake csv file
    header = ["Image Index", "Finding Labels"]

    with open(data_folder / "Data_Entry_2017_v2020.csv", "w", encoding="UTF8") as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the metadata for 100000 images
        for i in range(0, 5000):
            data = data = [str(i) + ".png", "No Finding"]
            writer.writerow(data)
        for i in range(5000, 8000):
            data = data = [str(i) + ".png", "Hernia|Emphysema|Pneumothorax"]
            writer.writerow(data)
        for i in range(8000, 9000):
            data = data = [str(i) + ".png", "Hernia"]
            writer.writerow(data)
        for i in range(9000, 10000):
            data = data = [
                str(i) + ".png",
                "Effusion|Emphysema|Infiltration|Pneumothorax",
            ]
            writer.writerow(data)
    yield data_folder

    shutil.rmtree(image_folder)
