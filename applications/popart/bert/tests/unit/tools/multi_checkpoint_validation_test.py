# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import logging
import random
import os
from pathlib import Path
from functools import partial, reduce
from operator import getitem
from contextlib import contextmanager
from bert import setup_logger
import tools.bert_multi_checkpoint_validation as validation

logging.getLogger().setLevel(logging.INFO)


@contextmanager
def does_not_raise():
    yield


def mock_infer_loop(f1, em, multiple_log_lines, correct_output, *params):
    dataset_logger = logging.getLogger("bert_data.squad_dataset")
    if multiple_log_lines:
        dataset_logger.error("log line 1")
        dataset_logger.warning("log line 2")
        dataset_logger.info("log line 3")
        dataset_logger.debug("log line 4")
    if correct_output:
        dataset_logger.info(f"F1 Score: {f1} | Exact Match: {em}")
    else:
        dataset_logger.info("Invalid output")


def mock_pooled_validation_run(checkpoint_paths,
                               root_path,
                               all_results,
                               bert_args,
                               config,
                               initializers,
                               paths_to_process,
                               num_processes=1,
                               available_ipus=16):
    results = validation.recursive_defaultdict()
    for path in checkpoint_paths:
        validation.result_into_recursive_path(
            results, path, root_path, all_results[path])
    return results


@pytest.mark.category1
@pytest.mark.parametrize(
    "multiple_log_lines, correct_output, expectation",
    [
        (True, True, does_not_raise()),
        (False, True, does_not_raise()),
        (False, False, pytest.raises(validation.NoResultRecordedException))
    ]
)
def test_run_inference_extract_result(monkeypatch, multiple_log_lines, correct_output, expectation):
    """Mock the output of the infer loop, and check the response against known-good values"""
    random.seed(1984)
    setup_logger(logging.getLevelName('INFO'))

    f1 = 100*random.random()
    em = 100*random.random()

    mock_func = partial(mock_infer_loop, f1, em,
                        multiple_log_lines, correct_output)
    monkeypatch.setattr(validation, "bert_infer_loop", mock_func)

    # Actual arguments passed in don't actually matter here as they're all passed straight down into run_infer_loop
    mock_args = [None]*7

    with expectation:
        j = validation.run_inference_extract_result(*mock_args)
        assert j["F1"] == f1
        assert j["exact_match"] == em


@pytest.mark.category1
@pytest.mark.parametrize(
    "root_path, ckpt_suffix, gt_indices",
    [
        ("/mock/root", "ckpt_0/run_0", ["ckpt_0", "run_0"]),
        ("/mock/root2", "ckpt_0/run_1", ["ckpt_0", "run_1"]),
        ("/mock/root3", "ckpt_0/run_2", ["ckpt_0", "run_2"])
    ]
)
def test_result_into_recursive_path(root_path, ckpt_suffix, gt_indices):
    """Test that the recursive parsing of path into a result dict works as intended.

    It should remove the common root leaving only a relative path to the testing
    checkpoint, then convert each layer of the path into a dict index.

    e.g.
        ckpt=/my/root/path/with/some/checkpoint.onnx, root=/my/root/path
        results = {
            "with": {
                "some": {
                    "checkpoint.onnx": {"my": "result"}
                }
            }
        }
    """
    random.seed(1984)
    result = {"f1": random.random()}
    ckpt_path = os.path.join(root_path, ckpt_suffix)
    result_dict = validation.recursive_defaultdict()

    validation.result_into_recursive_path(
        result_dict, ckpt_path, root_path, result)

    # Convert back to a dict so we don't inadvertently create indices while testing the result
    assert reduce(getitem, gt_indices, dict(result_dict)) == result


@pytest.mark.category1
@pytest.mark.parametrize(
    "num_experiments, num_runs, num_processes",
    [
        (3, 4, 1),
        (3, 4, 2),
        (3, 4, 3)
    ]
)
def test_perform_validations(monkeypatch, num_experiments, num_runs, num_processes):
    """ Test that the multi-process and single-process variants produce equivalent output
    dicts, given identical input. Mocks out the actual inference step."""
    random.seed(1984)

    def generate_result():
        return {
            "F1": 100*random.random(),
            "exact_match": 100*random.random()
        }

    root_path = "/mock/path"
    checkpoint_paths = [f"{root_path}/experiment_{e}/run_{s}/model.onnx"
                        for e in range(num_experiments) for s in range(num_runs)]

    all_results = {path: generate_result() for path in checkpoint_paths}

    mock_func = partial(mock_pooled_validation_run,
                        checkpoint_paths, root_path, all_results)
    monkeypatch.setattr(validation, "pooled_validation_run", mock_func)

    # args, config and initializers are all passed straight through to the pooled run, so they can be
    # set to None here.
    pooled_results = validation.perform_validations(num_processes,
                                                    checkpoint_paths,
                                                    None,
                                                    None,
                                                    None,
                                                    None)

    for path, expected_result in all_results.items():
        # Results use relative paths - get the relative path then traverse the recusive path dict
        # using it to get the result.
        relpath = os.path.relpath(path, root_path)
        assert expected_result == reduce(
            getitem, Path(relpath).parts, dict(pooled_results))


@pytest.mark.category1
@pytest.mark.parametrize(
    "num_processes,num_sweeps,num_repeats",
    [
        (3, 5, 5),
        (1, 5, 5),
        (3, 5, 1),
        (3, 1, 5),
    ]
)
def test_merge_hierarchy_results(num_processes, num_sweeps, num_repeats):
    """Ensure that nested directory paths are handled correctly by the validation result pooler"""

    # Generate a number of mocked paths that would go as input to validation
    paths = []
    root = '/mock/root/'
    path_pieces = [
        [f'sweep-{i}' for i in range(num_sweeps)],
        [f'repeat-{i}' for i in range(num_repeats)]
    ]
    for p in path_pieces[0]:
        for q in path_pieces[1]:
            paths.append(str(Path(root) / 'ckpt' / p / q))

    # Now pool them as we would if running pooled validation
    checkpoint_paths_pooled = (paths[i*len(paths) // num_processes:
                                     (i+1)*len(paths) // num_processes]
                               for i in range(num_processes))

    # Mock the results out (we don't care about the value)
    pooled_results = []
    for process_paths in checkpoint_paths_pooled:
        result_dict = validation.recursive_defaultdict()
        for path in process_paths:
            validation.result_into_recursive_path(
                result_dict, path, root, "A_MOCK_RESULT")
        pooled_results.append(result_dict)

    # Code under test
    mock_results = validation.merge_pooled_results(pooled_results)

    # Now check that the number of results provided in the test code exactly
    # matches the number of results given above.
    def count_leaves(d):
        count = 0
        for i in d.values():
            if isinstance(i, dict):
                count += count_leaves(i)
            else:
                count += 1
        return count

    assert count_leaves(mock_results) == len(paths)


if __name__ == "__main__":
    pytest.main(args=[__file__, '-vv', '-s'])
