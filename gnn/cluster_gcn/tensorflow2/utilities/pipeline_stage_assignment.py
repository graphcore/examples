# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from collections import defaultdict
import inspect
import logging

from tensorflow.python import ipu

from utilities.ipu_utils import get_matmul_options


class PipelineStagesAssigner:
    def __init__(self, pipeline_allocate_previous, pipeline_names):
        self.pipeline_allocate_previous = pipeline_allocate_previous
        self.pipeline_names = pipeline_names

    def assign_pipeline_stages(self, assignments, pipeline_stages):
        self.check_valid_model_expansion(assignments)
        num_layers_per_stage = self.count_pipeline_stages(pipeline_stages)
        pipeline_indexes = {i: key for key, val in self.pipeline_names.items() for i in val}

        stage_id = 0
        for i, a in enumerate(assignments):
            stage_id = self.find_next_available_stage(a.layer, num_layers_per_stage, pipeline_indexes, stage_id)
            a.pipeline_stage = stage_id

        # Check if there are any hidden layers in pipeline stages that
        # have been requested for allocation but don't exist in the model.
        for i, stage_slots in enumerate(num_layers_per_stage):
            if stage_slots.get("hid", 0) > 0:
                raise Exception("More hidden layers in pipeline assignment"
                                f" than in model on stage {i}.")

        return assignments

    def check_valid_model_expansion(self, assignments):
        valid_names = tuple(i for p in self.pipeline_names.values() for i in p if isinstance(i, str))
        valid_types = tuple(i for p in self.pipeline_names.values() for i in p if inspect.isclass(i))
        for assignment in assignments:
            if assignment.layer.name not in valid_names \
                    and not isinstance(assignment.layer, valid_types) \
                    and assignment.layer.name not in self.pipeline_allocate_previous \
                    and not isinstance(assignment.layer, self.pipeline_allocate_previous):
                raise ValueError(f"Layer with class {type(assignment.layer)} with name {assignment.layer.name} "
                                 f"is not in the valid pipeline classes.")
        return None

    @staticmethod
    def count_pipeline_stages(pipeline_stages):
        # Currently only the count of hidden layers is needed.
        num_layers_per_stage = list()
        for stage in pipeline_stages:
            num_layers_current = defaultdict(int)
            for name in stage:
                num_layers_current[name] += 1
            num_layers_per_stage.append(num_layers_current)
        return num_layers_per_stage

    def find_next_available_stage(self, layer, num_layers_per_stage, pipeline_index, stage_id):
        logging.debug(f"Looking for next available stage for layer {layer}")
        if layer.name in self.pipeline_allocate_previous or type(layer) in self.pipeline_allocate_previous:
            logging.debug(f"Layer {layer} should be added to same stage as previously"
                          f" allocated layer, stage {stage_id}")
            return stage_id

        for i, stage_slots in enumerate(num_layers_per_stage):
            valid_elements = [i for k in stage_slots.keys() for i in self.pipeline_names[k]]
            valid_names = tuple(name for name in valid_elements if isinstance(name, str))
            valid_types = tuple(name for name in valid_elements if inspect.isclass(name))
            if (layer.name in valid_names or (isinstance(layer, valid_types)) and pipeline_index[type(layer)] != "hid"):
                logging.debug(f"Layer {layer} isn't a hidden layer, allocating to stage {i}.")
                return i
            if isinstance(layer, valid_types) and pipeline_index[type(layer)] == "hid" and stage_slots["hid"] > 0:
                num_layers_per_stage[i]["hid"] -= 1
                logging.debug(f"Layer {layer} is a hidden layer, allocating to stage {i}.")
                return i
        raise Exception(f"No available slot for layer `{layer.name}` of type `{type(layer)}` "
                        f"with given pipeline stages.")


def get_poplar_options_per_pipeline_stage(num_ipus_per_replica,
                                          device_mapping,
                                          matmul_available_memory_proportion,
                                          matmul_partials_type):
    """
    Returns a list of poplar options per pipeline stage suitable
    for input into pipelining options.
    :param num_ipus_per_replica: An int representing the number of
        IPUs in the pipeline.
    :param device_mapping: A list of ints where each index is the pipeline
        stage and the value is the device ID that stage should be on.
    :param matmul_available_memory_proportion: A list of floats which
        determines the matmul available memory proportion for that
        IPU in the pipeline.
    :param matmul_partials_type: A string which determines type of the
        intermediate calculations of the matmuls. If using pipelining,
        this will be overridden by the setting per pipeline stage.
    :return: A list of PipelineStageOptions suitable for using in pipelining
        options.
    """
    if len(matmul_available_memory_proportion) != num_ipus_per_replica:
        raise ValueError(
            "Available memory proportion must be set for each of"
            f" the {num_ipus_per_replica} IPUs in the pipeline.")

    return [
        ipu.pipelining_ops.PipelineStageOptions(
            matmul_options=get_matmul_options(matmul_available_memory_proportion[stage],
                                              matmul_partials_type)
        ) for stage in device_mapping]


def pipeline_model(model,
                   config,
                   pipeline_names,
                   pipeline_allocate_previous,
                   num_ipus_per_replica,
                   matmul_partials_type):
    """
    Returns the provided model with a new pipeline stage assignment.
    :param config: A config namespace object that contains details about
        the dataset split configuration, see `utilities/options.py`.
    :param pipeline_names: Dictionary of user friendly names matching
        layer identifiers, which can be the layer name or its class.
    :param pipeline_allocate_previous: A list of layers classes that
        will be assigned to the same pipeline stage as the previous layer.
    :param num_ipus_per_replica: An int representing the number of IPUs
        used in a pipeline.
    :return: The model pipelined.
    """
    pipeline_assigner = PipelineStagesAssigner(pipeline_allocate_previous,
                                               pipeline_names)
    assignments = model.get_pipeline_stage_assignment()
    assignments = pipeline_assigner.assign_pipeline_stages(assignments,
                                                           config.ipu_config.pipeline_stages)

    # Apply the modified assignments back to the model.
    model.set_pipeline_stage_assignment(assignments)
    poplar_options_per_pipeline_stage = get_poplar_options_per_pipeline_stage(
        num_ipus_per_replica,
        config.ipu_config.pipeline_device_mapping,
        config.ipu_config.matmul_available_memory_proportion_per_pipeline_stage,
        matmul_partials_type
    )
    model.set_pipelining_options(
        gradient_accumulation_steps_per_replica=config.gradient_accumulation_steps_per_replica,
        device_mapping=config.ipu_config.pipeline_device_mapping,
        forward_propagation_stages_poplar_options=poplar_options_per_pipeline_stage,
        backward_propagation_stages_poplar_options=poplar_options_per_pipeline_stage,
        recomputation_mode=ipu.pipelining_ops.RecomputationMode.RecomputeAndBackpropagateInterleaved,
    )
    model.print_pipeline_stage_assignment_summary(print_fn=logging.info)
