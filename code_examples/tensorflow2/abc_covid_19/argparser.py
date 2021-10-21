# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""Parse command line arguments for the COVID-19 ABC modelling benchmark."""
import argparse


def get_argparser():
    """Argument parser for ABC algorithm parameters."""
    ap = argparse.ArgumentParser(
        'Approximate Bayesian Compute for COVID-19 Modelling')
    ap.add_argument('-t',
                    '--tolerance',
                    type=float,
                    required=False,
                    default=2e5,
                    help='ABC acceptance tolerance.')
    ap.add_argument('-s',
                    '--n-samples-target',
                    type=int,
                    required=False,
                    default=100,
                    help='Target number of approx. posterior samples.')
    ap.add_argument('-n',
                    '--n-days',
                    type=int,
                    required=False,
                    help='How many days to run inference for. '
                         'Reverts to number of days with data if not specified.')
    ap.add_argument('-cn',
                    '--country',
                    type=str,
                    required=False,
                    default='Italy',
                    choices=('Italy', 'New_Zealand', 'US'),
                    help="Which country's data to run ABC inference for.")
    ap.add_argument('-m',
                    '--max-n-runs',
                    type=int,
                    required=False,
                    default=100000000,
                    help='Maximum number of passes over the data before '
                         'terminating, even if num samples less than target.')
    ap.add_argument('-b',
                    '--n-samples-per-batch',
                    type=int,
                    required=False,
                    default=400000,
                    help='Number of samples to draw in each batch.')
    ap.add_argument('-r',
                    '--replication-factor',
                    type=int,
                    required=False,
                    default=1,
                    help='Number of IPU replicas.')
    ap.add_argument('-a',
                    '--sparse-output',
                    action='store_true',
                    help='Suppress most of the console output. Only total '
                         'time and time per run will be printed if specified.')
    ap.add_argument('-c',
                    '--enqueue-chunk-size',
                    type=int,
                    required=False,
                    help='Splits the batch of samples into chunks of this size, '
                         'and only sends the chunk to host if one or more of the '
                         'chunk are "accepted" samples.')
    ap.add_argument('-o',
                    '--outfeed-num-samples',
                    action='store_true',
                    help='Additionally to param vectors and their distances, '
                         'also outfeed how many samples were accepted '
                         'on each IPU.')
    ap.add_argument('-f',
                    '--no-outfeed-ops',
                    action='store_true',
                    help='If specified, no outfeed ops will be added to graph.')
    ap.add_argument('-fp',
                    '--samples-filepath',
                    required=False,
                    type=str,
                    help='Filepath to store accepted parameter samples. '
                         'Samples will NOT be saved if this arg not given.')
    return ap
