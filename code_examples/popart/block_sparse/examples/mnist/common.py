# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import subprocess

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import check_data_exists


def download_mnist(file_path):
    """Download the MNIST dataset hosted on Graphcore public S3 bucket"""

    if check_all_data_present(file_path):
        print("MNIST dataset already downloaded, skipping download")
        return

    subprocess.check_output(
        "./get_data.sh", env=os.environ.copy(), cwd=file_path,
        universal_newlines=True
    )

    if not check_all_data_present(file_path):
        raise OSError("MNIST dataset download unsuccessful")

    print("Successfully downloaded MNIST dataset")


def check_all_data_present(file_path):
    """Checks the data exists in location file_path"""

    filenames = [
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
    ]

    data_path = os.path.join(file_path, "data")

    return check_data_exists(data_path, filenames)
