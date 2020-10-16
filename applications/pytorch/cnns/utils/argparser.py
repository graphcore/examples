# Copyright 2020 Graphcore Ltd.
import argparse
import sys
sys.path.append('..')
import models  # noqa: E402


def get_common_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for training')
    parser.add_argument('--model', choices=models.available_models.keys(),  default='resnet18', help="Choose model")
    parser.add_argument('--pipeline-splits', type=str, nargs='+', default=[], help="List of the splitting layers")
    parser.add_argument('--replicas', type=int, default=1, help="Number of IPU replicas")
    parser.add_argument('--device-iteration', type=int, default=1, help="Device Iteration")
    parser.add_argument('--precision', choices=['full', 'half'], default='half', help="Floating Point precision")
    parser.add_argument('--half-partial', action='store_true', help='Accumulate matrix multiplication partials in half precision')
    parser.add_argument('--normlayer', choices=['batch', 'group', 'none'], default='batch',  help="Set normalization layers in the model")
    parser.add_argument('--groupnorm-group-num', type=int, default=32, help="In case of group normalization, the number of groups")
    parser.add_argument('--available-memory-proportion', type=float, default=[], nargs='+',
                        help='Proportion of memory which is available for convolutions. Use a value of less than 0.6')
    return parser
