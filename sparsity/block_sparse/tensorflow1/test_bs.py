# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
from bsmatmul_test import bs_matmul_test
from bssoftmax_test import bs_softmax_test
import utils

# test_data_train tuple --> (tag, scenario, data_type, partial_data_type, lhs_shape, rhs_shape (right 2 dimensions), block_size, sparsity, transposed_rhs, inner_group_size)
test_matmul_params = [
    # 2D
    #  inference
    #   float
    ("tag_mm_00", "dsd", "float", "float", [8, 8], [8, 8], [8, 8, 8], 0.0, False, 1),
    ("tag_mm_01", "dds", "float", "float", [8, 8], [8, 8], [8, 8, 8], 0.0, False, 1),
    #   half
    ("tag_mm_02", "dsd", "half", "half", [16, 16], [16, 16], [16, 16, 16], 0.0, False, 1),
    ("tag_mm_03", "dds", "half", "half", [16, 16], [16, 16], [16, 16, 16], 0.0, False, 1),
    #   half, partials float
    ("tag_mm_04", "dsd", "half", "float", [16, 16], [16, 16], [16, 16, 16], 0.0, False, 1),
    ("tag_mm_05", "dds", "half", "float", [16, 16], [16, 16], [16, 16, 16], 0.0, False, 1),
    # training
    ("tag_mm_08", "dsd-grad", "float", "float", [8, 8], [8, 8], [8, 8, 8], 0.0, False, 1),
    ("tag_mm_09", "dds-grad", "float", "float", [8, 8], [8, 8], [8, 8, 8], 0.0, False, 1),
    #  half
    ("tag_mm_10", "dsd-grad", "half", "half", [16, 16], [16, 16], [16, 16, 16], 0.0, False, 1),
    ("tag_mm_11", "dds-grad", "half", "half", [16, 16], [16, 16], [16, 16, 16], 0.0, False, 1),
    #   sparsity > 0
    ("tag_mm_12", "dsd-grad", "float", "float", [128, 64], [64, 32], [8, 8, 8], 0.5, False, 1),
    ("tag_mm_13", "dds-grad", "float", "float", [128, 64], [64, 32], [8, 8, 8], 0.5, False, 1),
    #   tranposed rhs
    ("tag_mm_14", "dsd-grad", "float", "float", [128, 64], [64, 32], [8, 8, 8], 0.5, True, 1),
    # 3D
    ("tag_mm_15", "dsd-grad", "half", "half", [3, 128, 32], [32, 64], [16, 16, 16], 0.9, False, 1),
    ("tag_mm_16", "dds-grad", "half", "half", [3, 128, 32], [32, 64], [16, 16, 16], 0.9, False, 1),
    # 4D
    ("tag_mm_17", "dsd-grad", "float", "float", [2, 2, 16, 32], [32, 16], [8, 8, 8], 0.8, True, 1),
    ("tag_mm_18", "dds-grad", "float", "float", [2, 2, 16, 32], [32, 16], [8, 8, 8], 0.8, False, 1),
    # Grouping
    #  partial grouping
    ("tag_mm_19", "dsd-grad", "float", "float", [2, 3, 16, 32], [32, 16], [8, 8, 8], 0.8, True, 3),
    ("tag_mm_20", "dds-grad", "float", "float", [2, 3, 16, 32], [32, 16], [8, 8, 8], 0.8, False, 2),
    #  full grouping
    ("tag_mm_21", "dsd-grad", "float", "float", [2, 3, 16, 32], [32, 16], [8, 8, 8], 0.8, False, 6),
    ("tag_mm_22", "dds-grad", "float", "float", [2, 3, 16, 32], [32, 16], [8, 8, 8], 0.8, False, 6),
    # Non-square block
    # dimension 0
    ("tag_mm_23", "dsd-grad", "float", "float", [128, 64], [64, 32], [32, 8, 8], 0.5, False, 1),
    ("tag_mm_24", "dsd-grad", "float", "float", [128, 64], [64, 32], [32, 8, 8], 0.5, True, 1),
    ("tag_mm_25", "dds-grad", "float", "float", [128, 64], [64, 32], [32, 8, 8], 0.5, False, 1),
    # dimension 1
    ("tag_mm_26", "dsd-grad", "float", "float", [128, 64], [64, 32], [8, 16, 8], 0.5, False, 1),
    ("tag_mm_27", "dsd-grad", "float", "float", [128, 64], [64, 32], [8, 16, 8], 0.5, True, 1),
    ("tag_mm_28", "dds-grad", "float", "float", [128, 64], [64, 32], [8, 16, 8], 0.5, False, 1),
    # dimension 2
    ("tag_mm_29", "dsd-grad", "float", "float", [128, 64], [64, 32], [8, 8, 16], 0.5, False, 1),
    ("tag_mm_30", "dsd-grad", "float", "float", [128, 64], [64, 32], [8, 8, 16], 0.5, True, 1),
    ("tag_mm_31", "dds-grad", "float", "float", [128, 64], [64, 32], [8, 8, 16], 0.5, False, 1),
    # all dimensions
    ("tag_mm_32", "dsd-grad", "float", "float", [2, 128, 64], [64, 32], [64, 32, 16], 0.5, False, 1),
    ("tag_mm_33", "dsd-grad", "float", "float", [2, 128, 64], [64, 32], [64, 32, 16], 0.5, True, 1),
    ("tag_mm_34", "dds-grad", "float", "float", [2, 128, 64], [64, 32], [64, 32, 16], 0.5, False, 1),
]


@pytest.mark.ipus(1)
@pytest.mark.parametrize("tag, scenario, data_type, partial_data_type, lhs_shape, rhs_shape, block_size, sparsity, transposed_rhs, inner_group_size", test_matmul_params)
def test_bsmatmul(tag, scenario, data_type, partial_data_type, lhs_shape, rhs_shape, block_size, sparsity, transposed_rhs, inner_group_size):

    class Opts(object):
        pass
    opts = Opts()
    opts.scenario = scenario
    opts.data_type = data_type
    opts.partial_data_type = partial_data_type
    opts.lhs_rows = lhs_shape[-2]
    opts.lhs_cols = lhs_shape[-1]
    opts.rhs_cols = rhs_shape[-1]
    opts.group_dims = lhs_shape[:-2]
    opts.lhs_block_row = block_size[0]
    opts.lhs_block_col = block_size[1]
    opts.rhs_block_col = block_size[2]
    opts.sparsity = sparsity
    opts.sparsity_mask = None
    opts.transposed_rhs = transposed_rhs
    opts.inner_group_size = inner_group_size
    opts.partition_method = "strip"
    opts.memory_cycle_ratio = 1.0

    out, lhs_grad, rhs_grad, out_ref, lhs_grad_ref, rhs_grad_ref = bs_matmul_test(opts)

    compute_grads = False
    if len(scenario) > 3:
        compute_grads = True

    if opts.data_type == "float":
        rtol = 1e-03
        atol = 1e-05
    else:
        rtol = 1e-01
        atol = 1e-03

    np.testing.assert_allclose(out, out_ref, rtol=rtol, atol=atol)
    if compute_grads:
        np.testing.assert_allclose(lhs_grad, lhs_grad_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(rhs_grad, rhs_grad_ref, rtol=rtol, atol=atol)

# test_data_train tuple --> (tag, data_type, shape, block_size, sparsity, sparsity_mask, inner_group_size, subblock_mask_type, in_place)
test_softmax_params = [
    # 2D
    #  Basic tests
    ("tag_sm_00", "float", [8, 8], [8, 8], 0.0, None, 1, ["no"], False),
    ("tag_sm_01", "half", [16, 16], [16, 16], 0.0, None, 1, ["no"], False),
    #  Sparsity
    ("tag_sm_02", "float", [64, 64], [8, 8], 0.5, None, 1, ["no"], False),
    ("tag_sm_03", "half", [64, 64], [16, 16], 0.5, None, 1, ["no"], False),
    # 3D
    ("tag_sm_04", "float", [3, 8, 8], [8, 8], 0.0, None, 1, ["no"], False),
    ("tag_sm_05", "half", [3, 16, 16], [16, 16], 0.0, None, 1, ["no"], False),
    #  Inner grouping
    ("tag_sm_06", "float", [6, 8, 8], [8, 8], 0.0, None, 2, ["no"], False),
    ("tag_sm_07", "half", [6, 16, 16], [16, 16], 0.0, None, 2, ["no"], False),
    # In place
    ("tag_sm_08", "float", [8, 8], [8, 8], 0.0, None, 1, ["no"], True),
    ("tag_sm_09", "half", [16, 16], [16, 16], 0.0, None, 1, ["no"], True),
    # Subblock mask
    ("tag_sm_10", "float", [8, 8], [8, 8], 0.0, None, 1, ["zut"], False),
    ("tag_sm_11", "float", [8, 8], [8, 8], 0.0, None, 1, ["zlt"], False),
    #  Different subblock mask for different groups
    ("tag_sm_12", "float", [2, 8, 8], [8, 8], 0.0, None, 1, ["zut", "zlt"], False),
    # Empty rows
    ("tag_sm_13", "float", [16, 16], [8, 8], None, "0011", 1, ["no"], False),
    # Non-square blocks
    ("tag_sm_14", "float", [16, 16], [16, 8], 0.0, None, 1, ["no"], False),
    # Empty rows as a result of subblock mask
    # [0|1] <- block mask
    # [0|1]
    #
    # [\|0] <- subblock mask: zero upper triangle
    # [1|\]
    #
    # [0|0] <- combination of 2 results in empty rows
    # [1|\]
    ("tag_sm_15", "float", [16, 16], [16, 8], None, "01", 1, ["zut"], False),
]


@pytest.mark.ipus(1)
@pytest.mark.parametrize("tag, data_type, shape, block_size, sparsity, sparsity_mask, inner_group_size, subblock_mask_type, in_place", test_softmax_params)
def test_bssoftmax(tag, data_type, shape, block_size, sparsity, sparsity_mask, inner_group_size, subblock_mask_type, in_place):

    class Opts(object):
        pass
    opts = Opts()
    opts.data_type = data_type
    opts.compute_grads = True
    opts.rows = shape[-2]
    opts.cols = shape[-1]
    opts.group_dims = shape[:-2]
    opts.block_row = block_size[0]
    opts.block_col = block_size[1]
    opts.sparsity = sparsity
    opts.sparsity_mask = sparsity_mask
    opts.inner_group_size = inner_group_size
    opts.subblock_mask_type = subblock_mask_type
    opts.in_place = in_place

    probs, loss, logits_grad, probs_ref, loss_ref, logits_grad_ref = bs_softmax_test(opts)

    if opts.data_type == "float":
        rtol = 1e-03
        atol = 1e-05
        rtol_grad = 1e-05
        atol_grad = 1e-03
    else:
        rtol = 1e-01
        atol = 1e-03
        rtol_grad = 2e-01
        atol_grad = 1e-03

    np.testing.assert_allclose(probs, probs_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(loss, loss_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(logits_grad, logits_grad_ref, rtol=rtol_grad, atol=atol_grad)
