# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict
import pandas as pd


def load_plot_data(filename):
    tensor_dict = {}

    """
    Format should be multiple lines with the following space separated fields:
    ITERATION RESULTS_1 RESULTS_2 ... RESULTS_N
    """
    with open(filename) as f:
        headers = f.readline().split()[1:]
        for line in f:
            tokens = line.split()
            itr = int(tokens[0])
            tensor_dict[itr] = [float(t) for t in tokens[1:]]

        return tensor_dict, headers


def convert_to_dataframes(data, colnames):
    frames = {}
    frames['values'] = pd.DataFrame.from_dict(data, orient='index', columns=colnames)
    return frames


def plot_data(frames, fields, args):
    plt.style.use(args.theme)

    # Compute how many individual lines will be plotted so we
    # can adjust the colormapping accordingly:
    count = len(fields)
    palette = plt.get_cmap(args.cmap, lut=int(np.round(count*1.25)))
    print(f"Plot count: {count}")

    # Do all the plotting:
    num = 0
    for id, f in frames.items():
        for field in fields:
            df = f[[field]]
            gap = df.index[-1] - df.index[-2]
            name = field
            for col in df:
                num += 1
                pcolor = palette(num % count)
                plt.plot(df[col], color=pcolor, linewidth=1, alpha=0.9, label=name)
                data_label_x = df.last_valid_index() + gap
                data_label_y = df[col][df[col].last_valid_index()]
                plt.text(data_label_x, data_label_y, name, horizontalalignment='left', size='x-small', color=pcolor)

    if args.logscale:
        plt.yscale('log')
    plt.xticks(rotation=55)
    plt.yticks(np.arange(0, 1.04, step=0.1))
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(args.title, loc='left', fontsize=12, color='black')
    plt.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and plot multiple runs on one chart')
    parser.add_argument('--input', help='Input test file in correct format', required=True, type=str)
    parser.add_argument('--output', help='Name for plot output. Format is derived from extension which must be a supported one.',
                        required=True, type=str)
    parser.add_argument('--theme', help='Matplotlib Theme.',
                        type=str, default='bmh', choices=plt.style.available)
    parser.add_argument('--cmap', help='Matplotlib Colormap.',
                        type=str, default='tab20', choices=plt.colormaps())
    parser.add_argument('--logscale', help='Set the y-axis use a logarithmic scale.', action='store_true')
    parser.add_argument('--title', help='Plot title', type=str, default="")
    parser.add_argument('--xlabel', help='X axis label', type=str, default="")
    parser.add_argument('--ylabel', help='Y axis label', type=str, default="")
    parser.add_argument('--headers', help='Override column headers from fiel with this list',
                        nargs='+', type=str, default=None)

    args = parser.parse_args()

    data, headers = load_plot_data(args.input)

    if args.headers is not None:
        if len(args.headers) != len(headers):
            raise ValueError("Length of header list provided on command line does "
                             "not match length of list from file.")
        headers = args.headers

    frames = convert_to_dataframes(data, headers)

    plot_data(frames, headers, args)
    plt.savefig(args.output)
