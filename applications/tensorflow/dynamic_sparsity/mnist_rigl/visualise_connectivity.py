# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


# Convert weights to a 0/1 mask:
def weights_to_mask(weights):
    mask = weights.copy()
    mask_idx = np.nonzero(mask)
    mask[mask_idx] = 1
    return mask


def mask_to_heat(mask):
    connection_sum = np.sum(mask, axis=1)
    return np.reshape(connection_sum, [28, 28])


# Plots the counts of hidden nodes connected to each input pixel:
def plot_mnist_connectivity(weights, title, axis, cbar_axis, args):
    heat = mask_to_heat(weights)
    axis.set_aspect('equal', 'box')
    if cbar_axis is None:
        g = sns.heatmap(heat, cmap=args.cmap, ax=axis)
    else:
        g = sns.heatmap(heat, cmap=args.cmap, ax=axis, cbar_ax=cbar_axis)
    g.set_title(title)
    g.set_xticks([])
    g.set_yticks([])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualise the connectivity in the network during training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cmap', help='Matplotlib Colormap.',
                        type=str, default='magma', choices=plt.colormaps())
    parser.add_argument("--records-path", type=str, required=True,
                        help="Path to masks and weights .npy files to visualise.")
    parser.add_argument("--animate", action='store_true',
                        help="If selected output frames for all files in records-path.")
    args = parser.parse_args()

    np_files = [f for f in os.listdir(args.records_path) if f.endswith('.npy')]
    np_files.sort()

    if not args.animate:
        print(f"Processing first and last record: {np_files[0]} {np_files[-1]}")

        grid = {'width_ratios': [1, 1, 0.08]}

        fig, axis = plt.subplots(1, 3, gridspec_kw=grid)
        axis[2].get_shared_y_axes().join(axis[0], axis[1])

        first_weights = np.load(os.path.join(args.records_path, np_files[0]))
        first_mask = weights_to_mask(first_weights)
        plot_mnist_connectivity(first_mask, 'Initial', axis[0], axis[2], args)

        last_weights = np.load(os.path.join(args.records_path, np_files[-1]))
        last_mask = weights_to_mask(last_weights)
        plot_mnist_connectivity(last_mask, 'Final', axis[1], axis[2], args)

        plt.savefig(os.path.join(args.records_path, "connectivity.png"))
    else:
        for i, f in enumerate(np_files):
            print(f"Plotting: {f}")
            weights = np.load(os.path.join(args.records_path, f))
            mask = weights_to_mask(weights)
            plt.clf()
            fig, ax = plt.subplots()
            plot_mnist_connectivity(mask, f"MNIST: Connections per pixel (rig-l steps: {i:02})", ax, None, args)
            plt.savefig(os.path.join(args.records_path, f"connectivity_{i:06}.png"))
            plt.close()
