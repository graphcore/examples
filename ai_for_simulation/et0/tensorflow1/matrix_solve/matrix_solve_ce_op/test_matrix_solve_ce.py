# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


import os
import numpy
import tensorflow as tf
import time
from tensorflow.python import ipu
from tensorflow.python.ipu import ipu_compiler, scopes, config

tf.compat.v1.disable_v2_behavior()


def matrix_solve_graph(A, b):
    outputs = {
        "output_types": [tf.float32],
        "output_shapes": [b.shape],
    }

    base_path = os.path.realpath(os.path.dirname(__file__))
    lib_path = os.path.join(base_path, "libmatrix_solve_ce_op.so")
    gp_path = os.path.join(base_path, "matrix_solve_codelets.gp")

    return ipu.custom_ops.precompiled_user_op([A, b], lib_path, gp_path, outs=outputs)


def main():
    print('kriging start')

    channel_cnt = numpy.int32(128*1024)
    loop_cnt = numpy.int32(100)
    slope_nugget = numpy.array([0.08119393835012971, 1.181032796509736], dtype=numpy.float32)
    gridx = numpy.arange(0.0, 6.0, 0.5, dtype=numpy.float32)
    gridy = numpy.arange(0.0, 6.0, 0.5, dtype=numpy.float32)
    x1, x2 = numpy.meshgrid(gridx, gridx)
    y1, y2 = numpy.meshgrid(gridy, gridy)
    dx = (x1 - x2)
    dy = (y1 - y2)
    d = numpy.sqrt(numpy.square(dx) + numpy.square(dy))
    kriging_mat = -(slope_nugget[0] * d + slope_nugget[1])

    kriging_mat_ext = numpy.pad(kriging_mat, ((0, 1), (0, 1)), 'constant', constant_values=1.0)
    diag_x = numpy.diag(numpy.diag(kriging_mat_ext))
    kriging_mat = kriging_mat_ext - diag_x

    for i in range(13):
        print('{:8f}f, {:8f}f, {:8f}f, {:8f}f, {:8f}f, {:8f}f, {:8f}f, {:8f}f, {:8f}f, {:8f}f, {:8f}f, {:8f}f, {:8f}f, '.format(kriging_mat[i][0], kriging_mat[i][1], kriging_mat[i][2], kriging_mat[i][3],
              kriging_mat[i][4], kriging_mat[i][5], kriging_mat[i][6], kriging_mat[i][7], kriging_mat[i][8], kriging_mat[i][9], kriging_mat[i][10], kriging_mat[i][11], kriging_mat[i][12]))

    kriging_mat = numpy.expand_dims(kriging_mat, axis=0)
    kriging_mat = numpy.repeat(kriging_mat, channel_cnt, axis=0)

    kriging_mat_cpy = numpy.copy(kriging_mat)
    print('kriging_mat_cpy.shape: {}'.format(kriging_mat_cpy.shape))

    b = numpy.array([1.2, 1.1, 1.3, 1.4, 1.6, 1.5, 1.7, 1.8, 1.9, 2.0, 2.1, 1.1, 1.4], dtype=numpy.float32)
    b = numpy.expand_dims(b, axis=0)
    b = numpy.expand_dims(b, axis=2)
    b = numpy.repeat(b, channel_cnt, axis=0)

    print('kriging_mat.shape: {}'.format(kriging_mat.shape))
    print('b.shape: {}'.format(b.shape))

    time_start = time.time()
    x = numpy.linalg.solve(kriging_mat, b)
    time_end = time.time()
    print('numpy total time elapsed: {}'.format((time_end - time_start)))
    print(x)

    connection_type = config.DeviceConnectionType.ALWAYS
    cfg = config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.device_connection.type = connection_type
    cfg.configure_ipu_system()
    with tf.device("cpu"):
        p_A = tf.compat.v1.placeholder(numpy.float32, kriging_mat.shape, name="A")
        p_b = tf.compat.v1.placeholder(numpy.float32, b.shape, name="b")

    with scopes.ipu_scope("/device:IPU:0"):
        mat_solve_model = ipu_compiler.compile(matrix_solve_graph, [p_A, p_b])

    with tf.compat.v1.Session() as sess:
        fd_solve = {p_A: kriging_mat, p_b: b}
        mat_solve_res = sess.run(mat_solve_model, fd_solve)
        print(mat_solve_res)
        time_start = time.time()
        for i in range(loop_cnt):
            fd_solve = {p_A: kriging_mat, p_b: b}
        mat_solve_res = sess.run(mat_solve_model, fd_solve)
        time_end = time.time()
        print('ipu total time elapsed: {}'.format((time_end - time_start)))
        cmp_diff = mat_solve_res[0] - x
        cmp_diff_abs = numpy.abs(cmp_diff)
        max_diff = numpy.max(cmp_diff_abs)
        min_diff = numpy.min(cmp_diff_abs)
        avg_diff = numpy.average(cmp_diff_abs)
        cmp_res = (cmp_diff_abs < 1e-6).all()
        print('cmp_res: {}, max_diff: {:.8f}, min_diff: {:.8f}, avg_diff: {:.8f}'.format(cmp_res, max_diff, min_diff, avg_diff))


if __name__ == "__main__":
    main()
