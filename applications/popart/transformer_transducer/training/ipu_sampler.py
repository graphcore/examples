# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
from common.data.dali.sampler import SimpleSampler, BucketingSampler, hash_list_of_strings


def make_file_list(sampler, output_files, json_names):
    suffix = ""
    if sampler.num_instances > 1:
        suffix = "." + str(sampler.instance_offset)
    # Each instance writes creates its own file list registry file
    sampler.file_list_path = os.path.join(
        "/tmp",
        "rnnt_dali.file_list." + hash_list_of_strings(json_names) + suffix
    )
    sampler.write_file_list(sampler.process_output_files(output_files))


class IpuBucketingSampler(BucketingSampler):
    """ This class is an adaptation of BucketingSampler so that it provides the data required for each poprun instance """
    def __init__(self, num_buckets, samples_per_step, num_epochs, rng, num_instances=1, instance_offset=0):
        super(IpuBucketingSampler, self).__init__(num_buckets, samples_per_step, 1, num_epochs, rng)
        self.num_instances = num_instances
        self.instance_offset = instance_offset

    def process_output_files(self, output_files):
        result = super(IpuBucketingSampler, self).process_output_files(output_files)
        if self.num_instances > 1:
            # Shard output file list between instances
            result_shard = []
            for idx, (name, label) in enumerate(result):
                if idx % self.num_instances == self.instance_offset:
                    result_shard.append((name, label))
            result = result_shard
        self.dataset_size = self.dataset_size // self.num_instances
        return result

    def make_file_list(self, output_files, json_names):
        make_file_list(self, output_files, json_names)


class IpuSimpleSampler(SimpleSampler):
    """ This class is an adaptation of SimpleSampler. It provides the data required for each poprun instance """

    def __init__(self, samples_per_step, num_instances=1, instance_offset=0):
        super(IpuSimpleSampler, self).__init__()
        self.num_instances = num_instances
        self.instance_offset = instance_offset
        self.samples_per_step = samples_per_step

    def process_output_files(self, output_files):
        data_elements = [item for item in output_files.items()]

        # Shard output file list between instances
        result = []
        for idx, (path, entry) in enumerate(data_elements):
            if idx % self.num_instances == self.instance_offset:
                result.append((path, entry['label']))
        self.dataset_size = len(result)
        return result

    def make_file_list(self, output_files, json_names):
        make_file_list(self, output_files, json_names)
