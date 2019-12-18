#!/usr/bin/env python3
import numpy as np
import argparse
import sys


def exponential_decay(s, args):
    p = args.parameters
    return p[0] * np.exp(-p[1]*s)


def linear_interp(s, args):
    p = args.parameters
    interp = (s - args.start)/(args.end - args.start)
    return p[0] + (interp * (p[1] - p[0]))


def cyclic_exponential_decay(s, args):
    p = args.parameters
    max_lr = p[0]
    period = p[1]
    decay_exponent = p[2]
    scale_a = p[3]
    scale_b = 1 - p[3]
    cyclic = np.abs(scale_a + scale_b * np.sin((s/period)*2*np.pi))
    decay = np.exp(-decay_exponent * s)
    return max_lr * cyclic * decay


def plot(s, fs, title, logscale, file_name):
    import matplotlib.pyplot as plt
    plt.style.use('classic')
    plt.plot(s, fs, 'x-')
    plt.xticks(s[::8], rotation=66)
    plt.grid(b=True, axis='both')
    plt.title(title)
    plt.xlabel('step')
    if logscale:
        plt.yscale('log')
        plt.ylabel('learning-rate (log-scale)')
    else:
        plt.ylabel('learning-rate')
    plt.savefig(file_name, bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    func_dict = {
        "exp": [exponential_decay, 2, "scale, exponent"],
        "linear": [linear_interp, 2, "first-value, last-value"],
        "cyclic-decay": [cyclic_exponential_decay, 4, "max-LR, step-period, exponent, scale"]
    }

    parser = argparse.ArgumentParser(description='Parameter Schedule Config Generator')
    parser.add_argument('--start', help='Starting step.',
                        type=int, default=0)
    parser.add_argument('--end', help='Ending step.',
                        type=int, default=21000)
    parser.add_argument('--interval', help='Step increment.',
                        type=int, default=512)
    parser.add_argument('--function', help='Function used to generate schedule',
                        type=str,
                        required=True,
                        choices=func_dict.keys())
    parser.add_argument('--parameters',
                        help='Parameters specific to each generator function',
                        type=float,
                        nargs='+',
                        default=[])
    parser.add_argument('--plot', help='Plot the schedule to a file', type=str, default=None)
    parser.add_argument('--logscale', help='Use a logarithmic scale when plotting.', action='store_true')
    args = parser.parse_args()

    # Check parameter count:
    expected_count = func_dict[args.function][1]
    if len(args.parameters) != expected_count:
        expected_description = func_dict[args.function][2]
        raise ValueError(
            f"Generator function '{args.function}' requires {expected_count} "
            f"parameters ({expected_description}).")

    func = func_dict[args.function][0]

    input = []
    output = []
    for s in range(args.start, args.end + args.interval, args.interval):
        input.append(s)
        output.append(func(s, args))

    comment = ' '.join(str(x) for x in sys.argv[1:])
    print(f'# Schedule Generator Arguments: {comment}')
    print('"lr_schedule_by_step": {')
    for s, fs in zip(input, output):
        print(f'    "{s}": {fs},')
    print('}')

    if args.plot:
        plot(input, output, comment, args.logscale, args.plot)
