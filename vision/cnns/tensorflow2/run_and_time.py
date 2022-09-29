# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
import os
import sys
import subprocess

'''
usage:

python3 run_and_time.py \
--num-replicas 16 --ipus-per-replica 1 --num-instances 8 \
--hosts pod_name1,pod_name2 --partition parition_name --vipu-host host_name \
--app-args='--config resnet50_mlperf_pod16_bs20 ...'
'''


def add_poprun_arguments(parser):
    # poprun parallelism
    parser.add_argument('--num-replicas', type=str, required=True,
                        help='Number of replicas for model parallelism, aka replication factor.')
    parser.add_argument('--ipus-per-replica', type=str, required=True,
                        help='Number for ipus for a replica, aka replica size.')
    parser.add_argument('--num-instances', type=str, required=True,
                        help='Number of instances for data parallelism.')

    # vipu args
    parser.add_argument('--hosts', type=str, required=True,
                        help='Host addresses separated by commas for mutli-host data parallelism.')
    parser.add_argument('--partition', type=str, required=True,
                        help='The name of the vipu partition.')
    parser.add_argument('--vipu-host', type=str, required=True,
                        help='The address of the vipu host machine.')

    # application arguments
    parser.add_argument('--app-args', default=None, type=str, required=False,
                        help='A string with arguments for the application.')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_poprun_arguments(parser)
    args = parser.parse_args()
    app_args = args.app_args.split(' ') if args.app_args is not None else []

    user = os.environ['USER']
    exec_cache = os.environ.get('TF_POPLAR_EXEC_CACHE') or os.path.join('/home', user, 'exec_cache')

    poprun_command = [
        'poprun', '-vv',
        '--host', args.hosts,
        '--only-output-from-instance', '0',
        '--mpi-global-args', '--mca oob_tcp_if_include eno1 --mca btl_tcp_if_include eno1 ',
        '--update-partition', 'yes',
        '--reset-partition', 'no',
        '--vipu-server-timeout', '600',
        '--vipu-server-host', args.vipu_host,
        '--vipu-partition', args.partition,
        '--executable-cache-path', exec_cache,
        '--num-instances', args.num_instances,
        '--num-replicas', args.num_replicas,
        '--ipus-per-replica', args.ipus_per_replica
    ]

    compilation_command = [
        *poprun_command,
        'python3', 'train.py', *app_args,
        '--mlperf-logging', 'True',
        '--num-epochs', '1',
        '--ckpts-per-epoch', '0',
        '--wandb', 'False'
    ]
    print(' '.join(compilation_command))

    # compile the training and validation model
    p = subprocess.Popen(compilation_command, stderr=sys.stderr, stdout=sys.stdout)
    p.wait()

    training_command = [
        *poprun_command,
        'python3', 'train.py', *app_args,
        '--mlperf-logging', 'True',
        '--wandb', 'True'
    ]
    print(' '.join(training_command))

    # run training
    p = subprocess.Popen(training_command, stderr=sys.stderr, stdout=sys.stdout)
    p.wait()
