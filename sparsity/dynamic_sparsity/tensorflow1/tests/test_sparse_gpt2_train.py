# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path

import datetime
import os
import sys

import pytest

build_dir = Path(__file__).parent.parent


@pytest.mark.usefixtures("ipu_sparse_ops", "wikitext_103_dataset")
class TestBuildAndRun(SubProcessChecker):

    def _run_command(self, args=""):
        cmd = sys.executable + " language_modelling/train_sparse.py"
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"

        # If CI_WANDB is set we allow wandb to sync CI results to
        # server otherwise we disable by setting WANDB_MODE to dryrun:
        enable_wandb = os.getenv('CI_WANDB')
        wandb_netrc = os.path.join(f"{os.getenv('HOME')}", '.netrc')
        if enable_wandb and os.path.isfile(wandb_netrc):
            print(f"Allowing wandb syncing during CI.")
        else:
            env["WANDB_MODE"] = "dryrun"

        dataset_dir = os.getenv(
            'CI_GPT2_DATASET',
            '/tmp/datasets/wikitext-103-gpt2/')

        self.run_command(f"{cmd} {args} --data-dir {dataset_dir} --on-demand",
                         build_dir,
                         [r"Throughput (\w+.\w+) token/s", r"Training complete."], env=env)

    @pytest.mark.skip(reason="needs wikitext103 dataset to run")
    @pytest.mark.ipus(4)
    @pytest.mark.ipu_version("ipu2")
    def test_rigl_train_gpt2_xl_4_ipus_3_encoders_wikitext103_wandb(self):
        date_str = datetime.date.today().isoformat()
        self._run_command(f'--use-wandb --wandb-project-name dynsparse-language-model --wandb-tags xl ci-testing --wandb-name gpt2-xl-train-ci-{date_str} --num-shards=4 --encoder-layers=3 --hidden-length=1600 --ff-length=6400 --source-vocab-length=20000 --warmup-steps=1000 --cooldown-steps=10000 --peak-learning-rate=0.01 --min-learning-rate=8e-6 --nepochs=2 --pipeline --repeat-count 100 --source-sequence-length 256 --gradient-accumulation-count 120 --dtype=float16 --block-size=16 --sparse-matmul-options={{"metaInfoBucketOversizeProportion":"0.2","partialsType":"half","availableMemoryProportion":"0.4"}} --encoder-stage-dense-matmul-options={{"availableMemoryProportion":"0.15"}} --dense-grad-matmul-options={{"availableMemoryProportion":"0.1","partialsType":"half"}} --sparsity=0.9 --pooling-type=SUM --prune-ratio 0 --grad-acculation-mode Avg --scale-grad-pre-acc --batch-size 1 --decay-power 0.9')

    @pytest.mark.skip(reason="needs wikitext103 dataset to run")
    @pytest.mark.ipus(2)
    @pytest.mark.ipu_version("ipu2")
    def test_rigl_train_gpt2_s_2_ipus_1_encoders_rigl_wikitext103_wandb(self):
        date_str = datetime.date.today().isoformat()
        self._run_command(f'--use-wandb --wandb-project-name dynsparse-language-model --wandb-tags s ci-testing rigl --wandb-name gpt2-s-rigl-train-ci-{date_str} --num-shards=2 --encoder-layers=1 --hidden-length=768 --ff-length=3072 --source-vocab-length=20000 --warmup-steps=1000 --cooldown-steps=10000 --peak-learning-rate=0.01 --min-learning-rate=8e-6 --nepochs=2 --pipeline --repeat-count 100 --source-sequence-length 256 --gradient-accumulation-count 120 --dtype=float16 --block-size=16 --sparse-matmul-options={{"metaInfoBucketOversizeProportion":"0.2","partialsType":"half","availableMemoryProportion":"0.4"}} --encoder-stage-dense-matmul-options={{"availableMemoryProportion":"0.15"}} --dense-grad-matmul-options={{"availableMemoryProportion":"0.1","partialsType":"half"}} --sparsity=0.9 --pooling-type=SUM --prune-ratio 0.5 --grad-acculation-mode Avg --scale-grad-pre-acc --batch-size 1 --decay-power 0.9')

    @pytest.mark.skip(reason="needs wikitext103 dataset to run")
    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_rigl_train_gpt2_s_1_ipu_1_encoders_rigl_wikitext103_wandb(self):
        date_str = datetime.date.today().isoformat()
        self._run_command(f'--use-wandb --wandb-project-name dynsparse-language-model --wandb-tags 1ipu ci-testing rigl --wandb-name gpt2-1ipu-rigl-train-ci-{date_str} --num-shards=1 --encoder-layers=1 --source-sequence-length 256 --hidden-length=768 --ff-length=3072 --source-vocab-length=30000 --repeat-count=10000 --warmup-steps=1000 --cooldown-steps=10000 --peak-learning-rate=2e-4 --min-learning-rate=8e-6 --nepochs=1 --prune-ratio 0.3 --block-size 16 --pooling-type=AVG --gradient-accumulation-count 60')
