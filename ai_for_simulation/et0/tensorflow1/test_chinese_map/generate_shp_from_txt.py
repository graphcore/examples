# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


import numpy as np


def process_prs(file):
    f = open(file)
    prs_list = []
    file_lines = f.readlines()
    for file_line in file_lines:
        file_split = file_line.split()
        name = file_split[0]
        x_temp = float(file_split[2])
        y_temp = float(file_split[1])
        x = np.floor(x_temp / 100) + \
            ((x_temp / 100 - np.floor(x_temp / 100)) * 100) / 60
        y = np.floor(y_temp / 100) + \
            ((y_temp / 100 - np.floor(y_temp / 100)) * 100) / 60
        if float(file_split[7]) < 20000:
            if float(file_split[3]) < 100000:
                attr_3 = float(file_split[3]) / 10
                attr_4 = float(file_split[7]) / 10
            else:
                attr_3 = (float(file_split[3]) - 100000) / 10
                attr_4 = float(file_split[7]) / 10
        else:
            if float(file_split[3]) < 100000:
                attr_3 = float(file_split[3]) / 10
                attr_4 = (float(file_split[7]) - 20000) / 10
            else:
                attr_3 = (float(file_split[3]) - 100000) / 10
                attr_4 = (float(file_split[7]) - 20000) / 10
        text = name + '\t' + str(x) + '\t' + str(y) + \
            '\t' + str(attr_3) + '\t' + str(attr_4) + '\n'
        prs_list.append(text)
    return prs_list


def process_rhu(file):
    f = open(file)
    rhu_list = []
    file_lines = f.readlines()
    for file_line in file_lines:
        file_split = file_line.split()
        if float(file_split[7]) < 32760:
            name = file_split[0]
            x_temp = float(file_split[2])
            y_temp = float(file_split[1])
            x = np.floor(x_temp / 100) + \
                ((x_temp / 100 - np.floor(x_temp / 100)) * 100) / 60
            y = np.floor(y_temp / 100) + \
                ((y_temp / 100 - np.floor(y_temp / 100)) * 100) / 60
            if float(file_split[3]) < 100000:
                attr_3 = float(file_split[3]) / 10
                attr_4 = float(file_split[7]) / 100
            else:
                attr_3 = (float(file_split[3]) - 100000) / 10
                attr_4 = float(file_split[7]) / 100
            text = name + '\t' + str(x) + '\t' + str(y) + \
                '\t' + str(attr_3) + '\t' + str(attr_4) + '\n'
            rhu_list.append(text)
    return rhu_list


def process_tem(file):
    f = open(file)
    tem_list = []
    file_lines = f.readlines()
    for file_line in file_lines:
        file_split = file_line.split()
        if float(file_split[7]) < 32760 and float(file_split[8]) < 32760 and float(file_split[9]) < 32760:
            name = file_split[0]
            x_temp = float(file_split[2])
            y_temp = float(file_split[1])
            x = np.floor(x_temp / 100) + \
                ((x_temp / 100 - np.floor(x_temp / 100)) * 100) / 60
            y = np.floor(y_temp / 100) + \
                ((y_temp / 100 - np.floor(y_temp / 100)) * 100) / 60
            if float(file_split[3]) < 100000:
                attr_3 = float(file_split[3]) / 10
            else:
                attr_3 = (float(file_split[3]) - 100000) / 10
            attr_4 = float(file_split[7]) / 10 + ((attr_3 / 100) * 0.65)
            attr_5 = float(file_split[8]) / 10 + ((attr_3 / 100) * 0.65)
            attr_6 = float(file_split[9]) / 10 + ((attr_3 / 100) * 0.65)
            text = name + '\t' + str(x) + '\t' + str(y) + '\t' + str(attr_3) + '\t' + str(
                attr_4) + '\t' + str(attr_5) + '\t' + str(attr_6) + '\n'
            tem_list.append(text)
    return tem_list


def process_win(file):
    f = open(file)
    win_list = []
    file_lines = f.readlines()
    for file_line in file_lines:
        file_split = file_line.split()
        if float(file_split[7]) < 32760:
            name = file_split[0]
            x_temp = float(file_split[2])
            y_temp = float(file_split[1])
            x = np.floor(x_temp / 100) + \
                ((x_temp / 100 - np.floor(x_temp / 100)) * 100) / 60
            y = np.floor(y_temp / 100) + \
                ((y_temp / 100 - np.floor(y_temp / 100)) * 100) / 60
            if float(file_split[7]) < 1000:
                if float(file_split[3]) < 100000:
                    attr_3 = float(file_split[3]) / 10
                    attr_4 = float(file_split[7]) / 10
                else:
                    attr_3 = (float(file_split[3]) - 100000) / 10
                    attr_4 = float(file_split[7]) / 10
            else:
                if float(file_split[3]) < 100000:
                    attr_3 = float(file_split[3]) / 10
                    attr_4 = (float(file_split[7]) - 1000) / 10
                else:
                    attr_3 = (float(file_split[3]) - 100000) / 10
                    attr_4 = (float(file_split[7]) - 1000) / 10
            text = name + '\t' + str(x) + '\t' + str(y) + \
                '\t' + str(attr_3) + '\t' + str(attr_4) + '\n'
            win_list.append(text)
    return win_list
