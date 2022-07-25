# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2015-2021, PyKrige Developers. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# Neither the name of PyKrige nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This file has been modified by Graphcore Ltd.


import os
import numpy
import tensorflow as tf
import time
import argparse
import ET0_generate
import generate_shp_from_txt
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from tensorflow.python import ipu
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ipu import ipu_compiler, scopes, config
from tensorflow.python.framework import errors
from libtiff import TIFF

tf.compat.v1.disable_v2_behavior()


def Generate_Init_target(sample_x, sample_y, cell_size, orignal_width, orignal_height):
    x_min = numpy.min(sample_x)
    x_max = numpy.max(sample_x)
    y_min = numpy.min(sample_y)
    y_max = numpy.max(sample_y)
    x_size = x_max - x_min
    y_size = y_max - y_min
    x_cell_cnt = numpy.int32(numpy.round(x_size / cell_size))
    y_cell_cnt = numpy.int32(numpy.round(y_size / cell_size))
    target_idx_x_i = numpy.arange(0, orignal_width, dtype=numpy.int32)
    target_idx_y_i = -numpy.arange(0, orignal_height, dtype=numpy.int32)
    diff_x = (orignal_width - x_cell_cnt) / 2
    diff_y = (orignal_height - y_cell_cnt) / 2
    x_min_new = numpy.max((0.0, x_min - diff_x * cell_size))
    y_max_new = y_max + diff_y * cell_size
    target_idx_x_f = target_idx_x_i * cell_size + x_min_new
    target_idx_y_f = y_max_new + target_idx_y_i * cell_size
    ptr_x, ptr_y = numpy.meshgrid(target_idx_x_f, target_idx_y_f)
    xpts = numpy.reshape(ptr_x, (target_idx_x_f.shape[0] * target_idx_y_f.shape[0], 1))
    ypts = numpy.reshape(ptr_y, (target_idx_x_f.shape[0] * target_idx_y_f.shape[0], 1))
    return target_idx_x_f, target_idx_y_f, xpts, ypts


def _adjust_for_anisotropy(X, center, scaling = [1.0], angle = [0.0]):
    center = numpy.asarray(center)[None, :]
    angle = numpy.asarray(angle) * numpy.pi / 180
    X -= center
    stretch = numpy.array([[1, 0], [0, scaling[0]]])
    rot_tot = numpy.array(
        [
            [numpy.cos(-angle[0]), -numpy.sin(-angle[0])],
            [numpy.sin(-angle[0]), numpy.cos(-angle[0])],
        ]
    )
    X_adj = numpy.dot(stretch, numpy.dot(rot_tot, X.T)).T
    X_adj += center
    return X_adj


def Adjust_Src_Point(src_x, src_y):
    XCENTER = (numpy.amax(src_x) + numpy.amin(src_x)) / 2.0
    YCENTER = (numpy.amax(src_y) + numpy.amin(src_y)) / 2.0
    X_ADJUSTED, Y_ADJUSTED = _adjust_for_anisotropy(
        numpy.vstack((src_x, src_y)).T,
        [XCENTER, YCENTER],
    ).T
    return XCENTER, YCENTER, X_ADJUSTED, Y_ADJUSTED


def Adjust_Dst_Point(src_center_x, src_center_y, dst_x, dst_y):
    pt_x, pt_y = _adjust_for_anisotropy(
        numpy.vstack((dst_x, dst_y)).T,
        [src_center_x, src_center_y]
    ).T
    return pt_x, pt_y


def Generate_Input_by_kDTree(src_x, src_y, target_pos_x, target_pos_y, nearest_points):
    xy_data = numpy.concatenate((src_x[:, numpy.newaxis], src_y[:, numpy.newaxis]), axis=1)
    xy_points = numpy.concatenate((numpy.squeeze(target_pos_x, axis=-1)[:, numpy.newaxis], numpy.squeeze(target_pos_y, axis=-1)[:, numpy.newaxis]), axis=1)
    print('xy_data.shape: {}'.format(xy_data.shape))
    print('xy_points.shape: {}'.format(xy_points.shape))
    tree = cKDTree(xy_data)
    bd, bd_idx = tree.query(xy_points, k=nearest_points, eps=0.0, n_jobs=8)
    bd = numpy.reshape(bd, (xy_points.shape[0] * nearest_points, ))
    full_idx = numpy.reshape(bd_idx, (xy_points.shape[0] * nearest_points, ))
    return full_idx, bd


def generate_distance_matrix(x, y, z):
    x1, x2 = numpy.meshgrid(x, x)
    y1, y2 = numpy.meshgrid(y, y)
    z1, z2 = numpy.meshgrid(z, z)
    dx = (x1 - x2)
    dy = (y1 - y2)
    d = numpy.sqrt(numpy.square(dx) + numpy.square(dy))
    g = 0.5 * numpy.square(z1 - z2)
    return d, g


def construct_fit_matrix(dist, semivar, nlags):
    indices = numpy.indices(dist.shape)
    dist = dist[(indices[0, :, :] > indices[1, :, :])]
    semivar = semivar[(indices[0, :, :] > indices[1, :, :])]

    dmax = numpy.amax(dist)
    dmin = numpy.amin(dist)
    dd = (dmax - dmin) / nlags
    bins = [dmin + n * dd for n in range(nlags)]
    dmax += 0.001
    bins.append(dmax)

    lags = numpy.zeros(nlags)
    semivariance = numpy.zeros(nlags)

    for n in range(nlags):
        if dist[(dist >= bins[n]) & (dist < bins[n + 1])].size > 0:
            lags[n] = numpy.mean(dist[(dist >= bins[n]) & (dist < bins[n + 1])])
            semivariance[n] = numpy.mean(semivar[(dist >= bins[n]) & (dist < bins[n + 1])])
        else:
            lags[n] = numpy.nan
            semivariance[n] = numpy.nan
    lags = lags[~numpy.isnan(semivariance)]
    semivariance = semivariance[~numpy.isnan(semivariance)]
    return lags, semivariance


def linear_variogram_model(m, d):
    slope = float(m[0])
    nugget = float(m[1])
    return slope * d + nugget


def _variogram_residuals(params, x, y, variogram_function, weight):
    resid = variogram_function(params, x) - y
    return resid


def FitLinearModel(x_data, y_data):
    x0 = [(numpy.amax(y_data) - numpy.amin(y_data)) / (numpy.amax(x_data) - numpy.amin(x_data)), numpy.amin(y_data), ]
    bnds = ([0.0, 0.0], [numpy.inf, numpy.amax(y_data)])
    res = least_squares(_variogram_residuals,
                        x0,
                        bounds=bnds,
                        loss="soft_l1",
                        args=(x_data, y_data, linear_variogram_model, False),)
    return res.x


def matrix_solve_ce_op(A, b):
    outputs = {
        "output_types":  [tf.float32],
        "output_shapes": [b.shape],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "../matrix_solve/matrix_solve_ce_op/libmatrix_solve_ce_op.so")
    gp_path = os.path.join(base_path, "../matrix_solve/matrix_solve_ce_op/matrix_solve_codelets.gp")

    x = ipu.custom_ops.precompiled_user_op([A, b], lib_path, gp_path, outs=outputs)
    return x


def EstimateData_inv_graph(grid_2_std_dist, std_x, std_y, std_z, slope_nugget):
    d0 = -grid_2_std_dist
    lamda0 = slope_nugget[0] * d0 + slope_nugget[1]
    lamda0 = tf.pad(lamda0, [[0, 0], [0, 1]], mode = 'CONSTANT', constant_values=1.0)
    lamda0 = tf.expand_dims(lamda0, -1)

    std_x2 = tf.repeat(std_x, std_x.shape[1], axis=1)
    std_x2 = tf.reshape(std_x2, (std_x.shape[0], std_x.shape[1], std_x.shape[1]))
    std_x1 = tf.transpose(std_x2, perm=[0, 2, 1])
    std_y2 = tf.repeat(std_y, std_y.shape[1], axis=1)
    std_y2 = tf.reshape(std_y2, (std_y.shape[0], std_y.shape[1], std_y.shape[1]))
    std_y1 = tf.transpose(std_y2, perm=[0, 2, 1])
    std_dx = (std_x1 - std_x2)
    std_dy = (std_y1 - std_y2)
    std_d0 = tf.sqrt(tf.square(std_dx) + tf.square(std_dy))
    std_kriging_matrix = -(slope_nugget[0] * std_d0 + slope_nugget[1])

    kriging_result_ext = tf.pad(std_kriging_matrix, [[0, 0], [0, 1], [0, 1]], mode = 'CONSTANT', constant_values=1.0)
    diag_x = array_ops.matrix_diag(array_ops.matrix_diag_part(kriging_result_ext))
    kriging_matrix = kriging_result_ext - diag_x

    solve_x = matrix_solve_ce_op(kriging_matrix, lamda0)
    lamda = solve_x[0]
    lamda = tf.squeeze(lamda, axis=2)
    lamda = tf.slice(lamda, [0, 0], [lamda0.shape[0], lamda0.shape[1] - 1])
    result = tf.reduce_sum(lamda*std_z, 1)
    return result


def temp_add(file_dem, data_temp):
    data_dem = Image.open(file_dem)
    data_dem = numpy.asarray(data_dem)
    temp = data_temp - 0.6 * data_dem / 100
    return temp


def main(prs_file, rhu_file, tem_file, win_file, height, width, file_dem, file_latit, tif_save):
    print('kriging start')

    time_shp_start = time.time()
    PRS_file = generate_shp_from_txt.process_prs(prs_file)
    RHU_file = generate_shp_from_txt.process_rhu(rhu_file)
    TEM_file = generate_shp_from_txt.process_tem(tem_file)
    WIN_file = generate_shp_from_txt.process_win(win_file)

    time_prc_start = time.time()
    length = len(PRS_file)
    X_ORIG = numpy.zeros(length, dtype=numpy.float32)
    Y_ORIG = numpy.zeros(length, dtype=numpy.float32)
    Z_ORIG_PRS = numpy.zeros(length, dtype=numpy.float32)
    Z_ORIG_RHU = numpy.zeros(length, dtype=numpy.float32)
    Z_ORIG_WIN = numpy.zeros(length, dtype=numpy.float32)
    Z_ORIG_TEM_AVG = numpy.zeros(length, dtype=numpy.float32)
    Z_ORIG_TEM_MIN = numpy.zeros(length, dtype=numpy.float32)
    Z_ORIG_TEM_MAX = numpy.zeros(length, dtype=numpy.float32)
    for index in range(length):
        prs_rec = PRS_file[index].split('\t')
        rhu_rec = RHU_file[index].split('\t')
        tem_rec = TEM_file[index].split('\t')
        win_rec = WIN_file[index].split('\t')
        X_ORIG[index] = float(prs_rec[1])
        Y_ORIG[index] = float(prs_rec[2])
        Z_ORIG_PRS[index] = float(prs_rec[4])
        Z_ORIG_RHU[index] = float(rhu_rec[4])
        Z_ORIG_WIN[index] = float(win_rec[4])
        Z_ORIG_TEM_AVG[index] = float(tem_rec[4])
        Z_ORIG_TEM_MAX[index] = float(tem_rec[5])
        Z_ORIG_TEM_MIN[index] = float(tem_rec[6])
    print('X_ORIG.shape {}'.format(X_ORIG.shape))
    print('Y_ORIG.shape {}'.format(Y_ORIG.shape))

    time_target_start = time.time()
    cell_size = 0.008333333333
    estimate_blk_size = 20000
    target_grid_x, target_grid_y, target_pos_x, target_pos_y = Generate_Init_target(X_ORIG, Y_ORIG, cell_size, width, height)
    estimate_blk_cnt = target_pos_x.shape[0] // estimate_blk_size
    estimate_blk_tail = target_pos_x.shape[0] - estimate_blk_cnt * estimate_blk_size
    print('target_pos.shape: {}, {}, estimate_blk_cnt {}, estimate_blk_tail {}'.format(target_pos_x.shape, target_pos_y.shape, estimate_blk_cnt, estimate_blk_tail))
    nearest_points = numpy.int32(12)

    kdTree_start = time.time()
    src_x_center, src_y_center, src_x_adjust, src_y_adjust = Adjust_Src_Point(X_ORIG, Y_ORIG)
    target_x_adjust, target_y_adjust = Adjust_Dst_Point(src_x_center, src_y_center, numpy.squeeze(target_pos_x, axis=1), numpy.squeeze(target_pos_y, axis=1))
    target_x_adjust = numpy.expand_dims(target_x_adjust, axis=-1)
    target_y_adjust = numpy.expand_dims(target_y_adjust, axis=-1)

    full_idx, dist_arr = Generate_Input_by_kDTree(src_x_adjust, src_y_adjust, target_x_adjust, target_y_adjust, nearest_points)
    x_origin_arr = src_x_adjust[full_idx]
    y_origin_arr = src_y_adjust[full_idx]
    kdTree_end = time.time()
    print("kDTree Cost: {}".format(kdTree_end - kdTree_start))

    time_fit_start = time.time()
    z_dict = {'Z_ORIG_PRS': Z_ORIG_PRS, 'Z_ORIG_RHU': Z_ORIG_RHU, 'Z_ORIG_WIN': Z_ORIG_WIN, 'Z_ORIG_TEM_AVG': Z_ORIG_TEM_AVG,
              'Z_ORIG_TEM_MAX': Z_ORIG_TEM_MAX, 'Z_ORIG_TEM_MIN': Z_ORIG_TEM_MIN}
    slope_nugget_dict = {}
    z_origin_arr_dict = {}
    for k, v in z_dict.items():
        Z_ORIG = v
        z_origin_arr_dict[k] = Z_ORIG[full_idx]
        lags_cnt = 12
        dist_mat_start = time.time()
        dist_mat, var_mat = generate_distance_matrix(src_x_adjust, src_y_adjust, Z_ORIG)
        dist_mat_end = time.time()

        coef_fit_start = time.time()
        coef_a, coef_b = construct_fit_matrix(dist_mat, var_mat, lags_cnt)
        slope_nugget_dict[k] = FitLinearModel(coef_a, coef_b)
        coef_fit_end = time.time()
        print('coef_fit cost: {}'.format(coef_fit_end - coef_fit_start))

    time_run_before_start = time.time()
    connection_type = config.DeviceConnectionType.ALWAYS
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.device_connection.type = connection_type
    cfg.configure_ipu_system()
    with tf.device("cpu"):
        pSa = tf.compat.v1.placeholder(numpy.float32, [estimate_blk_size, nearest_points], name="x")
        pSb = tf.compat.v1.placeholder(numpy.float32, [estimate_blk_size, nearest_points], name="y")
        pSc = tf.compat.v1.placeholder(numpy.float32, [estimate_blk_size, nearest_points], name="z")
        pslope_nugget = tf.compat.v1.placeholder(numpy.float32, [2, ], name="slope_neggut")
        p_test_dist_blk = tf.compat.v1.placeholder(numpy.float32, [estimate_blk_size, nearest_points], name="dist")

    with scopes.ipu_scope("/device:IPU:0"):
        if estimate_blk_cnt > 0:
            estimate_blk_model = ipu_compiler.compile(EstimateData_inv_graph, [p_test_dist_blk, pSa, pSb, pSc, pslope_nugget])

    with tf.compat.v1.Session() as sess:
        k = 0
        blk_start = k * estimate_blk_size * nearest_points
        cur_dist = dist_arr[blk_start:blk_start+estimate_blk_size * nearest_points]
        cur_std_input_x = x_origin_arr[blk_start:blk_start+estimate_blk_size * nearest_points]
        cur_std_input_y = y_origin_arr[blk_start:blk_start+estimate_blk_size * nearest_points]
        z_origin_arr = z_origin_arr_dict['Z_ORIG_PRS']
        cur_std_input_z = z_origin_arr[blk_start:blk_start+estimate_blk_size * nearest_points]
        cur_dist = numpy.reshape(cur_dist, (estimate_blk_size, nearest_points))
        cur_std_input_x = numpy.reshape(cur_std_input_x, (estimate_blk_size, nearest_points))
        cur_std_input_y = numpy.reshape(cur_std_input_y, (estimate_blk_size, nearest_points))
        cur_std_input_z = numpy.reshape(cur_std_input_z, (estimate_blk_size, nearest_points))
        festimate_blk = {p_test_dist_blk: cur_dist, pSa: cur_std_input_x, pSb: cur_std_input_y, pSc: cur_std_input_z, pslope_nugget: slope_nugget_dict['Z_ORIG_PRS']}
        time_get_data_end = time.time()

        estimate_blk_result = sess.run(estimate_blk_model, festimate_blk)

        time_start = time.time()
        total_result = {}
        for key in z_origin_arr_dict.keys():
            result = []
            slope_nugget = slope_nugget_dict[key]
            z_origin_arr = z_origin_arr_dict[key]
            for k in range(estimate_blk_cnt + 1):
                blk_start = k * estimate_blk_size * nearest_points
                if k != estimate_blk_cnt:
                    cur_dist = dist_arr[blk_start:blk_start+estimate_blk_size * nearest_points]
                    cur_std_input_x = x_origin_arr[blk_start:blk_start+estimate_blk_size * nearest_points]
                    cur_std_input_y = y_origin_arr[blk_start:blk_start+estimate_blk_size * nearest_points]
                    cur_std_input_z = z_origin_arr[blk_start:blk_start+estimate_blk_size * nearest_points]
                else:
                    zeros = numpy.zeros((estimate_blk_size * nearest_points), dtype=numpy.float32)
                    zeros[:estimate_blk_tail*nearest_points] = dist_arr[blk_start:]
                    cur_dist = zeros
                    zeros[:estimate_blk_tail*nearest_points] = x_origin_arr[blk_start:]
                    cur_std_input_x = zeros
                    zeros[:estimate_blk_tail*nearest_points] = y_origin_arr[blk_start:]
                    cur_std_input_y = zeros
                    zeros[:estimate_blk_tail*nearest_points] = z_origin_arr[blk_start:]
                    cur_std_input_z = zeros
                cur_dist = numpy.reshape(cur_dist, (estimate_blk_size, nearest_points))
                cur_std_input_x = numpy.reshape(cur_std_input_x, (estimate_blk_size, nearest_points))
                cur_std_input_y = numpy.reshape(cur_std_input_y, (estimate_blk_size, nearest_points))
                cur_std_input_z = numpy.reshape(cur_std_input_z, (estimate_blk_size, nearest_points))
                festimate_blk = {p_test_dist_blk: cur_dist, pSa: cur_std_input_x, pSb: cur_std_input_y, pSc: cur_std_input_z, pslope_nugget: slope_nugget}
                estimate_blk_result = sess.run(estimate_blk_model, festimate_blk)
                result.append(estimate_blk_result)
                if 0 == (k % 64):
                    time_end = time.time()
                    print('time elapsed: {}'.format(time_end - time_start))
                    print('finished blk {}'.format(k))
            time_end = time.time()
            print('total time elapsed: {}'.format(time_end - time_start))

            res = numpy.array(result).flatten()
            res = res[:target_grid_y.shape[0]*target_grid_x.shape[0]]
            res = numpy.reshape(res, (target_grid_y.shape[0], target_grid_x.shape[0]))

            if 'TEM' in key:
                res = temp_add(file_dem, res)
            total_result[key] = res
            time_tem_add_end = time.time()

    time_et0_calc_start = time.time()
    img_wind = total_result['Z_ORIG_WIN']
    img_humi = total_result['Z_ORIG_RHU']
    img_pres = total_result['Z_ORIG_PRS']
    img_temp_mean = total_result['Z_ORIG_TEM_AVG']
    img_temp_max = total_result['Z_ORIG_TEM_MAX']
    img_temp_min = total_result['Z_ORIG_TEM_MIN']
    temp_result_et = ET0_generate.calc_et0(img_wind, img_humi, img_pres, img_temp_mean, img_temp_max, img_temp_min, file_latit, file_dem)
    time_et0_calc_end = time.time()

    im = Image.fromarray(temp_result_et)
    tif = TIFF.open(tif_save, mode='w')
    tif.write_image(im, compression=None)

    cost_shp = time_prc_start - time_shp_start
    cost_prc = time_target_start - time_prc_start
    cost_target = kdTree_start - time_target_start
    cost_kdTree = kdTree_end - kdTree_start
    cost_fit = time_run_before_start - time_fit_start
    cost_run_before = time_get_data_end - time_run_before_start
    cost_complie = time_start - time_get_data_end
    cost_kriging = time_tem_add_end - time_start
    cost_et0_cal = time_et0_calc_end - time_et0_calc_start
    cost_add_all = cost_shp + cost_prc + cost_target + cost_kdTree + cost_fit + cost_run_before + cost_kriging + cost_et0_cal
    cost_total = time_et0_calc_end - time_shp_start

    print('cost_shp: {}'.format(cost_shp))
    print('cost_prc: {}'.format(cost_prc))
    print('cost_target: {}'.format(cost_target))
    print('cost_kdTree: {}'.format(cost_kdTree))
    print('cost_fit: {}'.format(cost_fit))
    print('cost_run_before: {}'.format(cost_run_before))
    print('cost_kriging: {}'.format(cost_kriging))
    print('cost_et0_cal: {}'.format(cost_et0_cal))
    print('cost_add_all: {}'.format(cost_add_all))
    print('cost_total: {}'.format(cost_total))
    print('cost_complie: {}'.format(cost_complie))
    print('Complete the calculation of et0 in 1 day')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='build kdTree')
    parser.add_argument('--prs-file', type=str, default='dummy_data/dummy_raw_data/SURF_CLI_CHN_MUL_DAY-PRS-10004-20210403.TXT', help='path of the PRS file')
    parser.add_argument('--rhu-file', type=str, default='dummy_data/dummy_raw_data/SURF_CLI_CHN_MUL_DAY-RHU-13003-20210403.TXT', help='path of the RHU file')
    parser.add_argument('--tem-file', type=str, default='dummy_data/dummy_raw_data/SURF_CLI_CHN_MUL_DAY-TEM-12001-20210403.TXT', help='path of the TEM file')
    parser.add_argument('--win-file', type=str, default='dummy_data/dummy_raw_data/SURF_CLI_CHN_MUL_DAY-WIN-11002-20210403.TXT', help='path of the WIN file')
    parser.add_argument('--height', type=int, default=4534, help='height of the output file(tif format)')
    parser.add_argument('--width', type=int, default=7346, help='width of the output file(tif format)')
    parser.add_argument('--file-dem', type=str, default='dummy_data/dummy_resource/dummy_dem.tif', help='path of the DEM file')
    parser.add_argument('--file-latit', type=str, default='dummy_data/dummy_latitude/dummy_lati.tif', help='path of the latitude file')
    parser.add_argument('--tif-save', type=str, default='test_result/ET0_generate_1.tif', help='save path of the output file in tif type')
    args = parser.parse_args()
    main(args.prs_file, args.rhu_file, args.tem_file, args.win_file, args.height, args.width, args.file_dem, args.file_latit, args.tif_save)
