# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pytest
import os
import torch
import numpy as np
import poptorch
from torchvision import transforms
from pathlib import Path
import shutil
import random
import import_helper
from datasets.augmentations import AugmentationModel
from datasets.preprocess import IgnoreBboxIfPresent, NormalizeToTensor, RandomResizedFlipCrop, get_preprocessing_pipeline
from datasets.webdataset_format import DistributeNode, match_preprocess
from datasets.dataset import get_data, _WorkerInit
from datasets.create_webdataset import parse_transforms
import models
from models.models import NormalizeInputModel
from utils import run_script, get_current_interpreter_executable


class TestCustomAugmentations():
    @staticmethod
    def deterministic_get_params(*args):
        return 20, 20, 100, 100

    @staticmethod
    def _generate_img(size=256):
        torch.manual_seed(0)
        tensor = torch.rand(3, size, size)
        to_pil, to_tensor = transforms.ToPILImage(), transforms.ToTensor()
        img = to_pil(tensor)
        tensor = to_tensor(img) * 255.0  # Make sure the image is the same
        return img, tensor

    def test_custom_flip(self):
        """
        Compares original and fast flip.
        """
        tensor = torch.rand(1, 3, 100, 100)
        custom = RandomResizedFlipCrop.fast_hflip(tensor)
        correct = transforms.functional.hflip(tensor)
        assert torch.allclose(custom, correct, atol=1e-06)

    def test_NormalizeToTensor_pil(self):
        """
        Check the fused normalise+ToTensor steps against the original.
        """
        img, tensor = TestCustomAugmentations._generate_img()
        custom_pipeline = NormalizeToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        correct_pipeline = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        custom = custom_pipeline(img)
        correct = correct_pipeline(img)
        assert torch.allclose(custom, correct, atol=1e-06)

    def test_RandomResizedFlipCrop_pil(self):
        """
        Check the fused RandomResizedCrop+RandomFlip steps against the separate one.
        """
        img, tensor = TestCustomAugmentations._generate_img()
        correct_pipeline = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        custom_pipeline = transforms.Compose([RandomResizedFlipCrop(224), transforms.ToTensor()])
        RandomResizedFlipCrop.get_params = self.deterministic_get_params
        transforms.RandomResizedCrop.get_params = self.deterministic_get_params
        torch.manual_seed(0)
        custom = custom_pipeline(img)
        torch.manual_seed(0)
        correct = correct_pipeline(img)
        assert torch.allclose(custom, correct, atol=1e-06)

    def test_inference_pipeline(self):
        """
        Compare original validation pipeline against the optimised one.
        """
        img, tensor = TestCustomAugmentations._generate_img(256)
        ground_truth_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        preprocess = get_preprocessing_pipeline(train=False, input_size=224, half_precision=False, normalize=True)
        result_img = preprocess(img)
        result_tensor = preprocess(tensor)
        ground_truth = ground_truth_preprocess(img)
        assert torch.allclose(result_img, ground_truth, atol=1e-06)  # reference vs custom(img)
        assert torch.allclose(result_img, result_tensor, atol=1e-06)  # custom(img) vs custom(tensor)

    def test_fine_tuning_pipeline_ignores_bbox_if_present(self):
        preprocess = get_preprocessing_pipeline(train=True, fine_tuning=True)
        assert isinstance(preprocess.transforms[0], IgnoreBboxIfPresent)
        img, tensor = TestCustomAugmentations._generate_img(256)
        # Just make sure processing with bbox doesn't result in an error.
        preprocess((img, "bbox"))
        preprocess(img)
        preprocess((tensor, "bbox"))
        preprocess(tensor)


    def test_RandomResizedFlipCrop_tensor(self):
        img = torch.randint(low=0, high=255, size=(3, 256, 256), dtype=torch.uint8)
        correct_pipeline = transforms.Compose([transforms.ToPILImage(),
                                               transforms.RandomResizedCrop(100),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        custom_pipeline = transforms.Compose([RandomResizedFlipCrop(100), NormalizeToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        RandomResizedFlipCrop.get_params = self.deterministic_get_params
        transforms.RandomResizedCrop.get_params = self.deterministic_get_params
        torch.manual_seed(0)
        custom = custom_pipeline(img)
        torch.manual_seed(0)
        correct = correct_pipeline(img)
        assert torch.allclose(custom, correct, atol=1e-06)

    def test_ipu_side_normalization(self):
        img_host = torch.rand(3, 100, 100) * 255.0
        img_ipu = img_host.clone()
        host_pipeline = NormalizeToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model = models.NormalizeInputModel(lambda x: x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], output_cast="full")
        host_result = host_pipeline(img_host)
        ipu_result = model(img_ipu)
        assert torch.allclose(host_result, ipu_result, atol=1e-06)

    def test_mixup(self):
        class Model(torch.nn.Module):
            def forward(self, batch):
                return batch

        images = torch.stack([
            torch.ones(1, 12, 12),
            torch.ones(1, 12, 12) * 2,
            torch.ones(1, 12, 12) * 3,
        ]).to(torch.float)
        labels = torch.tensor([1, 2, 3])
        mixup_coeffs = torch.tensor([0.5, 0.5, 0.5])

        model = poptorch.inferenceModel(AugmentationModel(
            Model(),
            use_mixup=True,
            use_cutmix=False,
        ))
        mixed_images, all_coeffs = model((images, mixup_coeffs))

        correct_mixed_images = torch.stack([
            torch.ones(1, 12, 12) * 2,    # 0.5 * 1 + 0.5 * 3
            torch.ones(1, 12, 12) * 1.5,  # 0.5 * 2 + 0.5 * 1
            torch.ones(1, 12, 12) * 2.5,  # 0.5 * 3 + 0.5 * 2
        ]).to(torch.float)

        torch.testing.assert_allclose(mixed_images, correct_mixed_images)
        torch.testing.assert_allclose(all_coeffs[0], mixup_coeffs)

        all_labels, weights = model.mix_labels(labels, all_coeffs)
        torch.testing.assert_allclose(
            actual=all_labels[0],
            expected=labels,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_allclose(
            actual=all_labels[1],
            expected=torch.tensor([3, 1, 2]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_allclose(weights[0], mixup_coeffs)
        torch.testing.assert_allclose(weights[1], 1.0 - mixup_coeffs)

    def test_cutmix(self):
        class Model(torch.nn.Module):
            def forward(self, batch):
                return batch

        images = torch.stack([
            torch.ones(1, 10, 10),
            torch.ones(1, 10, 10) * 2,
            torch.ones(1, 10, 10) * 3,
        ]).to(torch.float)
        labels = torch.tensor([1, 2, 3])

        class args: pass
        args = args()
        # 0.75 should result in 5x5 cut boxes.
        args.cutmix_lambda_low = 0.75
        args.cutmix_lambda_high = 0.75
        args.cutmix_disable_prob = 0.0
        args.precision = "32.32"

        model = poptorch.inferenceModel(AugmentationModel(
            Model(),
            use_mixup=False,
            use_cutmix=True,
            args=args,
        ))
        cutmixed_images, all_coeffs = model((images, ()))

        torch.testing.assert_allclose(all_coeffs[0], torch.tensor(0.75))
        torch.testing.assert_allclose(
            torch.sum(cutmixed_images, dim=(1, 2, 3)),
            torch.tensor([
                0.75 * images[0].sum() + 0.25 * images[1].sum(),
                0.75 * images[1].sum() + 0.25 * images[2].sum(),
                0.75 * images[2].sum() + 0.25 * images[0].sum(),
            ]),
        )
        torch.testing.assert_allclose(
            actual=torch.unique(cutmixed_images[0], sorted=True),
            expected=torch.tensor([1.0, 2.0]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_allclose(
            actual=torch.unique(cutmixed_images[1], sorted=True),
            expected=torch.tensor([2.0, 3.0]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_allclose(
            actual=torch.unique(cutmixed_images[2], sorted=True),
            expected=torch.tensor([1.0, 3.0]),
            rtol=0,
            atol=0,
        )

        all_labels, weights = model.mix_labels(labels, all_coeffs)
        torch.testing.assert_allclose(
            actual=all_labels[0],
            expected=labels,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_allclose(
            actual=all_labels[1],
            expected=torch.tensor([2, 3, 1]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_allclose(weights[0], torch.tensor(0.75))
        torch.testing.assert_allclose(weights[1], 1.0 - torch.tensor(0.75))

    def test_cutmix_disabled_with_coefficient(self):
        class Model(torch.nn.Module):
            def forward(self, batch):
                return batch

        images = torch.stack([
            torch.ones(1, 10, 10),
            torch.ones(1, 10, 10) * 2,
            torch.ones(1, 10, 10) * 3,
        ]).to(torch.float)

        class args: pass
        args = args()
        # Value of 1.0 for lambda disables cutmix.
        args.cutmix_lambda_low = 1.0
        args.cutmix_lambda_high = 1.0
        args.cutmix_disable_prob = 0.0
        args.precision = "32.32"

        model = poptorch.inferenceModel(AugmentationModel(
            Model(),
            use_mixup=False,
            use_cutmix=True,
            args=args,
        ))
        cutmixed_images, all_coeff = model((images, ()))

        torch.testing.assert_allclose(all_coeff[0], torch.tensor(1.0))
        torch.testing.assert_allclose(cutmixed_images, images)


class TestHostBenchmark:
    def test_host_benchmark_cifar10(self):
        output = run_script("datasets/host_benchmark.py", "--data cifar10 --batch-size 256")
        assert "Throughput of the iteration" in output

    def test_poprun_host_benchmark(self):
        executable = get_current_interpreter_executable()
        output = run_script("poprun", f"--mpi-global-args='--allow-run-as-root' --num-instances=2 --numa-aware=yes --offline-mode=1 --num-replicas=2 {executable} datasets/host_benchmark.py --data cifar10 --batch-size 256", python=False)
        assert "Throughput of the iteration" in output


@pytest.mark.ipus(1)
@pytest.mark.parametrize("dataset", ["real", "generated"])
@pytest.mark.parametrize("precision", ["16.16", "32.32"])
def test_input_8bit(dataset, precision):
    """ Test 8-bit input vs usual input precision
    """
    def run_model(dl, eight_bit_io=False):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                return x * 2.0
        cast_op = "half" if precision == "16.16" else "full"
        model = NormalizeInputModel(MyModel(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], output_cast=cast_op) if eight_bit_io else MyModel()
        poptorch_model = poptorch.inferenceModel(model, poptorch.Options())
        input_data = next(iter(dl))[0]
        return poptorch_model(input_data)

    class HelperClass:
        def __init__(self):
            pass
    args = HelperClass()
    opts = poptorch.Options()
    args.batch_size = 1
    args.dataloader_worker = 4
    args.data = dataset
    args.model = "resnet18"
    args.precision = precision
    args.eight_bit_io = False
    args.normalization_location = "host"
    dataloader = get_data(args, opts, train=False)
    result_normal = run_model(dataloader, eight_bit_io=False)
    args.eight_bit_io = True
    args.normalization_location = "ipu"
    dataloader8 = get_data(args, opts, train=False)
    result_8bit = run_model(dataloader8, eight_bit_io=True)
    if not dataset == "generated":
        assert torch.allclose(result_8bit, result_normal, atol=4e-03, rtol=1e-03)
    assert result_normal.type() == result_8bit.type()


def test_chunk_distribution():
    """ Test the distributed webdataset, whether all the chunks are used.
    """
    nodes = []
    urls = ["chunk{}".format(i) for i in range(1251)]
    remaining = ["remain{}".format(i) for i in range(8)]
    for i in range(8):
        nodes.append(DistributeNode(remaining, i, 8))
    for epoch in range(10):
        splitted_url = []
        for node_id in range(8):
            splitted_url += nodes[node_id](urls)
        assert len(splitted_url) == 8 + 1248
        url_set = set(splitted_url)
        for i in range(1248):
            assert "chunk{}".format(i) in url_set
        for i in range(8):
            assert "remain{}".format(i) in url_set


class TestWebDataset:
    def test_webdataset_creation(self):
        raw_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("cifar10_raw")
        converted_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("test_cifar10_webdata_creation")
        # create train folder from validation
        if not os.path.exists(os.path.join(raw_path, "train")):
            shutil.copytree(os.path.join(raw_path, "validation"), os.path.join(raw_path, "train"))
        run_script("datasets/create_webdataset.py", f"--source {raw_path} --target {converted_path} --shuffle --seed 0 --format tensor --samples-per-shard 200")
        out = run_script("datasets/validate_dataset.py", f"--imagenet-data-path {converted_path}")
        num_files = len(os.listdir(converted_path))
        shutil.rmtree(converted_path)
        shutil.rmtree(os.path.join(raw_path, "train"))
        assert "Dataset OK." in out
        assert num_files == 2 * 50 + 1

    def test_webdataset_preprocess(self):
        pipeline = parse_transforms(["Resize(200)", "CenterCrop(100)"])
        assert len(pipeline) == 2
        assert isinstance(pipeline[0], type(transforms.Resize(200)))
        assert vars(pipeline[0]) == vars(transforms.Resize(200))
        assert isinstance(pipeline[1], type(transforms.CenterCrop(100)))
        assert vars(pipeline[1]) == vars(transforms.CenterCrop(100))


    def test_webdataset_distribution(self):
        """Smoke test for distributed webdataset generation.
        """
        webdata_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("cifar10_webdata")
        distributed_folder = os.path.join(webdata_path, "distributed", "8-instances")
        if os.path.exists(distributed_folder):
            shutil.rmtree(distributed_folder)
        run_script("datasets/distributed_webdataset.py", f"--target {webdata_path} --num-instances 8")
        assert os.path.exists(distributed_folder)
        num_files = len(os.listdir(distributed_folder))
        shutil.rmtree(distributed_folder)

        assert num_files == 2 * 8


    def test_webdata_cache(self):
        """Test cache for webdata
        """
        class HelperClass:
            def __init__(self):
                pass
        args = HelperClass()
        args.precision = "16.16"
        args.model = 'resnet50'
        args.device_iterations = 1
        args.replicas = 1
        args.batch_size = 31
        args.dataloader_worker = 8
        args.normalization_location = 'ipu'
        args.eight_bit_io = False
        args.webdataset_percentage_to_use = 100
        args.data = "imagenet"
        args.webdataset_memory_cache_ratio = 0.8
        args.imagenet_data_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("cifar10_webdata")

        dataloader = get_data(args, poptorch.Options(), train=False, async_dataloader=True, return_remaining=True)
        total_samples = 0
        for data, label in dataloader:
            total_samples += label.size()[0]
        assert total_samples == 10000


@pytest.mark.parametrize("first_step", ["", "Resize(200)", "Resize(256)"])
@pytest.mark.parametrize("second_step", ["", "CenterCrop(224)"])
def test_preprocess_match(first_step, second_step):
    """
    The webdataset may contain augmentations. Check wether the duplicated preprocessing steps are removed from the pipeline.
    """
    first_step = [] if first_step == "" else [first_step]
    second_step = [] if second_step == "" else [second_step]
    done_transform = first_step + second_step
    pipeline = get_preprocessing_pipeline(train=False, input_size=224, half_precision=False, normalize=True)
    len_pipeline = len(pipeline.transforms)
    modified_pipeline = match_preprocess(pipeline, done_transform)
    if len(first_step) == 0 or first_step[0] != "Resize(256)":
        assert len_pipeline == len(modified_pipeline.transforms)
    else:
        if len(second_step) == 0:
            assert len_pipeline - 1 == len(modified_pipeline.transforms)
        else:
            assert len_pipeline - 2 == len(modified_pipeline.transforms)


@pytest.mark.parametrize("async_dataloader", [True, False])
@pytest.mark.parametrize("return_remaining", [True, False])
@pytest.mark.parametrize("data_type", ["raw", "webdata"])
@pytest.mark.parametrize("num_instances", [1, 4])
def test_get_data(async_dataloader, return_remaining, data_type, num_instances):
    """
    Check whether all the samples are used.
    """
    class HelperClass:
        def __init__(self):
            pass
    args = HelperClass()
    args.precision = "16.16"
    args.model = 'resnet50'
    args.device_iterations = 1
    args.replicas = 1
    args.batch_size = 31
    args.dataloader_worker = 8
    args.normalization_location = 'ipu'
    args.eight_bit_io = False
    args.webdataset_percentage_to_use = 100
    if data_type == "webdata":
        args.data = "imagenet"
        args.imagenet_data_path = Path(__file__).parent.parent.absolute().joinpath("data").joinpath("cifar10_webdata")
    else:
        args.data = 'cifar10'
    lengths = []
    for instance_id in range(num_instances):
        opts = poptorch.Options()
        if num_instances > 1:
            opts.Distributed.configureProcessId(instance_id, num_instances)
        dataloader = get_data(args, opts, train=False, async_dataloader=async_dataloader, return_remaining=return_remaining)
        length = 0
        for x, y in dataloader:
            length += x.size()[0]
        lengths.append(length)
    if return_remaining:
        assert sum(lengths) == 10000
        assert len(dataloader) == 10000 // (num_instances * 31) + 1
    else:
        expected_batch_count = 10000 // (num_instances * 31)
        assert sum(lengths) == expected_batch_count * (num_instances * 31)
        assert len(dataloader) == expected_batch_count


@pytest.mark.parametrize("random_generator", ["numpy", "torch", "python"])
@pytest.mark.parametrize("instances", [1, 2])
def test_random_raw(random_generator, instances):
    """
    Tests whether all the augmentations are unique.
    """
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=10, transform=None):
            self.size = size
            self.transform = transform

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            if self.transform == "numpy":
                augment = np.random.random(1)[0]
            elif self.transform == "torch":
                augment = torch.rand(1)[0]
            elif self.transform == "python":
                augment = random.random()
            else:
                augment = 0.0
            return float(index) + augment

    ds = DummyDataset(transform=random_generator)
    augments = []
    elements = []
    for instance_id in range(instances):
        opts = poptorch.Options()
        worker_init = _WorkerInit(42, instance_id, 5)
        if instances > 1:
            opts.Distributed.configureProcessId(instance_id, instances)
        opts = opts.randomSeed(42)
        data_loader = poptorch.DataLoader(opts, ds, batch_size=1, num_workers=5, shuffle=True, worker_init_fn=worker_init)
        for item in data_loader:
            frac = item[0].numpy().tolist() % 1  # Get fraction(augmentation)
            frac = int(10000 * frac)  # avoid rounding error
            augments.append(frac)
            elements.append(int(item))
    assert len(elements) == len(set(elements))
    assert len(augments) == len(set(augments))  # all augmentations must be unique
