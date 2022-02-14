# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import os
import requests
import subprocess
import tarfile


def download_weights():
    weight_file = 'yolov4_p5_reference_weights'
    weight_folder = os.environ['PYTORCH_APPS_DETECTION_PATH'] + '/weights'
    file_path = Path(weight_folder)

    if not file_path.exists():
        file_path.mkdir()

    url = 'https://gc-demo-resources.s3.us-west-1.amazonaws.com/' + weight_file + '.tar.gz'
    target_path = weight_folder + '/' + weight_file + '.tar.gz'

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())

    if not Path(target_path).exists():
        raise Exception("The file " + target_path + " is missing")

    tar_weights_folder = tarfile.open(target_path)
    tar_weights = tar_weights_folder.getmember(weight_file + '/yolov4-p5-sd.pt')
    tar_weights.name = Path(tar_weights.name).name
    tar_weights_folder.extract(tar_weights, weight_folder)
    tar_weights_folder.close()


def pytest_sessionstart(session):
    path_to_detection = Path(__file__).parent.resolve()
    os.environ['PYTORCH_APPS_DETECTION_PATH'] = str(path_to_detection)
    subprocess.run(['make'], shell=True, cwd=path_to_detection)
    build_folder_path = path_to_detection.joinpath("utils/custom_ops/build")
    assert build_folder_path.is_dir()
    download_weights()
