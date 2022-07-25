# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


import numpy as np
from PIL import Image
from numpy.lib.twodim_base import _trilu_indices_form_dispatcher


def is_leapyear(year):
    if year % 100 == 0:
        if year % 400 == 0:
            return 366
        else:
            return 365
    else:
        if year % 4 == 0:
            return 366
        else:
            return 365


def cal_slope_vp(data_tem_mean):
    tem_slope_vp = (4098 * (0.6108 * np.exp((17.27 * data_tem_mean) / (data_tem_mean + 237.3)))) / np.square(data_tem_mean + 237.3)
    return tem_slope_vp


def cal_saturation_vp(data_tem_max, data_tem_min):
    temp_satur_vp = ((0.6108 * np.exp((17.27 * data_tem_max) / (data_tem_max + 237.3))) +
                     (0.6108 * np.exp((17.27 * data_tem_min) / (data_tem_min + 237.3)))) / 2
    return temp_satur_vp


def cal_actual_vp(data_humi, data_satur_vp):
    temp_actual_vp = data_humi * data_satur_vp
    return temp_actual_vp


def cal_rindex(data_press):
    temp_rindex = 0.664742 * 0.001 * (data_press / 10.0)
    return temp_rindex


def cal_raidation(data_latit, data_temmax, data_temmin, data_dem, data_vctual_vp, day):
    sun_number = 0.082
    data_latit = (np.pi / 180) * data_latit
    dr = 1 + 0.033 * np.cos(((2 * np.pi) / 365) * day)
    sun_latit = 0.409 * np.sin(((2 * np.pi) / 365) * day - 1.39)
    sun_time_angle = np.arccos(-np.tan(data_latit) * np.tan(sun_latit))
    temp_ra = 24.0 * 60.0 / np.pi * sun_number * dr * (sun_time_angle * np.sin(data_latit) * np.sin(sun_latit) +
                                                       np.cos(data_latit) * np.cos(sun_latit) * np.sin(sun_time_angle))
    temp_rs = (0.16 * np.sqrt(data_temmax - data_temmin)) * temp_ra
    temp_rso = (0.75 + 0.00002 * data_dem) * temp_ra
    temp_rns = 0.77 * temp_rs
    temp_rnl = 4.903e-9 * (((data_temmax + 273.16) ** 4 + (data_temmin + 273.16) ** 4) / 2) * \
                          (0.34 - 0.14 * data_vctual_vp ** 0.5) * (1.35 * (temp_rs / temp_rso) - 0.35)
    temp_rn = temp_rns - temp_rnl
    return temp_rn


def read_tif(file):
    im = Image.open(file)
    img = np.asarray(im)
    return img


def calc_et0(img_wind, img_humi, img_pres, img_temp_mean, img_temp_max, img_temp_min, file_latit, file_dem):
    img_latit = read_tif(file_latit)
    re = read_tif(file_dem)
    height, width = re.shape[0], re.shape[1]
    block_num = 30
    block_height = height // block_num

    file_name = 'SURF_CLI_CHN_MUL_DAY-WIN-11002-20210403.tif'
    tmp = file_name.split('-')
    year = int(tmp[3][:4])
    month = int(tmp[3][4:6])
    day = int(tmp[3][6:8])

    temp_result_et = np.zeros((height, width))
    a = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    b = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if month >= 2:
        if is_leapyear(year) == 365:
            i = np.sum(a[0:month - 2]) + day
        else:
            i = np.sum(b[0:month - 2]) + day
    else:
        i = day

    for block in range(0, block_num):
        if block == block_num - 1:
            h = height - block_height * block
        else:
            h = block_height
        data_dem = re[block * block_height:block*block_height + h, :]
        data_latit = img_latit[block * block_height:block*block_height + h, :]
        data_wind = img_wind[block * block_height:block*block_height + h, :]
        data_humi = img_humi[block * block_height:block*block_height + h, :]
        data_pres = img_pres[block * block_height:block*block_height + h, :]
        data_tem_mean = img_temp_mean[block * block_height:block*block_height + h, :]
        data_tem_max = img_temp_max[block * block_height:block*block_height + h, :]
        data_tem_min = img_temp_min[block * block_height:block*block_height + h, :]

        temp_result_slope_vp = cal_slope_vp(data_tem_mean)
        temp_result_satur_vp = cal_saturation_vp(data_tem_max, data_tem_min)
        temp_result_actual_vp = cal_actual_vp(data_humi, temp_result_satur_vp)
        temp_result_rindex = cal_rindex(data_pres)
        temp_result_rn = cal_raidation(data_latit, data_tem_max, data_tem_min, data_dem, temp_result_actual_vp, i)

        temp_result_et[block * block_height:block*block_height + h, :] = \
            (0.408 * temp_result_slope_vp * temp_result_rn + temp_result_rindex * (900.0 / (data_tem_mean + 273)) *
             data_wind * (temp_result_satur_vp - temp_result_actual_vp)) \
            / (temp_result_slope_vp + temp_result_rindex * (1 + 0.34 * data_wind))
    return temp_result_et
