# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import popdist.tensorflow


class BatchConfig:

    def __init__(self,
                 micro_batch_size,
                 num_clusters,
                 clusters_per_batch,
                 max_nodes_per_batch,
                 executions_per_epoch,
                 gradient_accumulation_steps_per_replica,
                 num_replicas,
                 epochs_per_execution=1,
                 distributed_training=False,
                 num_real_nodes_per_epoch=None,
                 num_epochs=1):
        self.micro_batch_size = micro_batch_size
        self.clusters_per_batch = clusters_per_batch
        self.max_nodes_per_batch = max_nodes_per_batch
        self.executions_per_epoch = executions_per_epoch
        self.gradient_accumulation_steps_per_replica = gradient_accumulation_steps_per_replica
        self.num_replicas = num_replicas
        self.num_real_nodes_per_epoch = num_real_nodes_per_epoch
        self.num_epochs = num_epochs
        self.epochs_per_execution = epochs_per_execution
        self.real_over_padded_ratio = None

        if self.epochs_per_execution > 1:
            logging.warning(
                "Epochs per execution has been set to"
                f" {self.epochs_per_execution} to improve the multi-instance performance."
                " The number of epochs run in model.fit"
                f" will be corrected to {self.scaled_num_epochs} to allow this.")

        if distributed_training:
            num_instances = popdist.getNumInstances()
        else:
            num_instances = 1
        # We are scaling one epoch up with epochs_per_execution, meaning that one epoch could train
        # more than one time of the whole training dataset.
        if distributed_training:
            all_instance_steps_per_epoch = num_clusters // self.clusters_per_batch * self.epochs_per_execution
        else:
            all_instance_steps_per_epoch = num_clusters // self.clusters_per_batch
        num_nodes_per_epoch = all_instance_steps_per_epoch * self.max_nodes_per_batch
        if self.num_real_nodes_per_epoch is not None:
            # Scale the number of real nodes up with epochs_per_execution to make a fair calculation
            self.real_over_padded_ratio = (self.num_real_nodes_per_epoch * self.epochs_per_execution) / num_nodes_per_epoch
        else:
            self.real_over_padded_ratio = 1.0

        # steps_per_epoch get truncated here to distribute to multiple replicas
        steps_per_execution_per_replica = max(
            1,
            all_instance_steps_per_epoch // (self.num_replicas *
                                             self.executions_per_epoch)
        )
        all_replica_steps_per_execution = steps_per_execution_per_replica * self.num_replicas

        new_all_replica_steps_per_execution = self.round_down_to_multiple(
            all_replica_steps_per_execution, self.gradient_accumulation_steps_per_replica)
        if new_all_replica_steps_per_execution != all_replica_steps_per_execution:
            logging.warning(
                "Steps per execution has been truncated from"
                f" {all_replica_steps_per_execution} to {new_all_replica_steps_per_execution}"
                " in order for it to be divisible by gradient accumulation"
                " steps per replica.")
        all_replica_steps_per_execution = new_all_replica_steps_per_execution

        new_all_instance_steps_per_epoch = all_replica_steps_per_execution * self.executions_per_epoch
        if new_all_instance_steps_per_epoch != all_instance_steps_per_epoch:
            logging.warning(
                "Steps per epoch has been truncated from"
                f" {all_instance_steps_per_epoch} to {new_all_instance_steps_per_epoch}"
                " in order for it to be divisible by steps per execution as required by the tensorflow API")
        all_instance_steps_per_epoch = new_all_instance_steps_per_epoch
        # As num_replicas should be divisible by num_instance
        # all_instance_steps_per_epoch is divisible by num_replicas, so it
        # is also divisible by num_instance
        self.steps_per_epoch = all_instance_steps_per_epoch // num_instances
        self.steps_per_execution = all_replica_steps_per_execution // self.num_replicas

    @property
    def global_batch_size(self):
        return (self.micro_batch_size *
                self.gradient_accumulation_steps_per_replica *
                self.num_replicas)

    @property
    def num_nodes_processed_per_execution(self):
        return (self.micro_batch_size *
                self.max_nodes_per_batch *
                self.steps_per_execution *
                self.num_replicas)

    @property
    def num_clusters_processed_per_execution(self):
        return (self.micro_batch_size *
                self.clusters_per_batch *
                self.steps_per_execution *
                self.num_replicas)

    @property
    def scaled_num_epochs(self):
        return max(1, self.num_epochs // self.epochs_per_execution)

    @staticmethod
    def round_down_to_multiple(in_val, round_to):
        return (in_val // round_to) * round_to

    def __str__(self):
        if self.epochs_per_execution > 1:
            self.epochs_per_execution_string = f"Epochs per execution: {self.epochs_per_execution}"
        else:
            self.epochs_per_execution_string = ""
        return (f"\tMicro batch size: {self.micro_batch_size}\n\t"
                f"Gradient accumulation count: {self.gradient_accumulation_steps_per_replica}\n\t"
                f"Global batch size: {self.global_batch_size}\n\t"
                f"Number of replicas: {self.num_replicas}\n\t"
                f"Steps per execution: {self.steps_per_execution}\n\t"
                f"Steps per epoch: {self.steps_per_epoch}\n\t"
                f"Max nodes per batch: {self.max_nodes_per_batch}\n\t"
                f"Clusters per batch: {self.clusters_per_batch}\n\t"
                f"Number of nodes processed per execution: {self.num_nodes_processed_per_execution}\n\t"
                f"Number of clusters processed per execution: {self.num_clusters_processed_per_execution}\n\t"
                f"Number of actual epochs: {self.num_epochs}\n\t"
                f"Number of corrected epochs (to account allow for multiple epochs in a single execution): {self.scaled_num_epochs}\n\t"
                f"{self.epochs_per_execution_string}")
