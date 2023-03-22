# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import inspect
import logging
from collections import defaultdict

from tensorflow.python import ipu
from utils import str_dtype_to_tf_dtype


def get_matmul_options(available_memory_proportion, matmul_partials_type):
    return {"availableMemoryProportion": str(available_memory_proportion), "partialsType": matmul_partials_type}


def get_pipeline_stages_equal_split(assignmemts, num_pipeline_stages, pipeline_allocate_previous):
    new_assignments = []
    for _, assignment in enumerate(assignmemts):
        # remove dummy layer from assignments
        if (
            assignment.layer.name not in pipeline_allocate_previous
            and type(assignment.layer) not in pipeline_allocate_previous
        ):
            new_assignments.append(assignment)
    num_layers = len(new_assignments)
    num_stages = num_layers // num_pipeline_stages
    remainder = num_layers - (num_stages * num_pipeline_stages)
    n_layers_per_stage = [num_stages] * num_pipeline_stages
    if remainder > 0:
        for i in range(remainder):
            n_layers_per_stage[num_pipeline_stages - (i + 1)] += 1

    stage_id_list = [i for i, n in enumerate(n_layers_per_stage) for _ in range(n)]
    device_mapping = list(set(stage_id_list))

    stage_id = 0
    for _, a in enumerate(assignmemts):
        stage_id = find_pipeline_stage(a.layer, stage_id, pipeline_allocate_previous, stage_id_list)
        a.pipeline_stage = stage_id
    return device_mapping


def find_pipeline_stage(layer, stage_id, pipeline_allocate_previous, stage_id_list):
    # if the layer is a dummy layer, assign it to the previous pipeline stage
    if layer.name in pipeline_allocate_previous or type(layer) in pipeline_allocate_previous:
        return stage_id
    else:
        stage_id = stage_id_list.pop(0)
        return stage_id


class PipelineStagesAssigner:
    def __init__(self, pipeline_allocate_previous, pipeline_names):
        self.pipeline_allocate_previous = pipeline_allocate_previous
        self.pipeline_names = pipeline_names

    def assign_pipeline_stages(self, assignments, pipeline_stages):
        self.check_valid_model_layers(assignments)
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
                raise Exception("More hidden layers in pipeline assignment" f" than in model on stage {i}.")

        return assignments

    def check_valid_model_layers(self, assignments):
        valid_names = tuple(i for p in self.pipeline_names.values() for i in p if isinstance(i, str))
        valid_types = tuple(i for p in self.pipeline_names.values() for i in p if inspect.isclass(i))
        for assignment in assignments:
            if (
                assignment.layer.name not in valid_names
                and not isinstance(assignment.layer, valid_types)
                and assignment.layer.name not in self.pipeline_allocate_previous
                and not isinstance(assignment.layer, self.pipeline_allocate_previous)
            ):
                raise ValueError(
                    f"Layer with class {type(assignment.layer)} with name {assignment.layer.name} "
                    f"is not in the valid pipeline classes."
                )
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
            logging.debug(
                f"Layer {layer} should be added to same stage as previously" f" allocated layer, stage {stage_id}"
            )
            return stage_id

        for i, stage_slots in enumerate(num_layers_per_stage):
            valid_elements = [i for k in stage_slots.keys() for i in self.pipeline_names[k]]
            valid_names = tuple(name for name in valid_elements if isinstance(name, str))
            valid_types = tuple(name for name in valid_elements if inspect.isclass(name))
            if layer.name in valid_names or (isinstance(layer, valid_types)) and pipeline_index[type(layer)] != "hid":
                logging.debug(f"Layer {layer} isn't a hidden layer, allocating to stage {i}.")
                return i
            if isinstance(layer, valid_types) and pipeline_index[type(layer)] == "hid" and stage_slots["hid"] > 0:
                num_layers_per_stage[i]["hid"] -= 1
                logging.debug(f"Layer {layer} is a hidden layer, allocating to stage {i}.")
                return i
        raise Exception(
            f"No available slot for layer `{layer.name}` of type `{type(layer)}` " f"with given pipeline stages."
        )


def get_poplar_options_per_pipeline_stage(
    num_ipus_per_replica, device_mapping, available_memory_proportion, matmul_partials_type
):
    """
    Returns a list of poplar options per pipeline stage suitable
    for input into pipelining options.
    :param num_ipus_per_replica: An int representing the number of
        IPUs in the pipeline.
    :param device_mapping: A list of ints where each index is the pipeline
        stage and the value is the device ID that stage should be on.
    :param available_memory_proportion: A list of floats which
        determines the matmul available memory proportion for that
        IPU in the pipeline.
    :param matmul_partials_type: A string which determines type of the
        intermediate calculations of the matmuls. If using pipelining,
        this will be overridden by the setting per pipeline stage.
    :return: A list of PipelineStageOptions suitable for using in pipelining
        options.
    """
    if len(available_memory_proportion) != num_ipus_per_replica:
        if len(available_memory_proportion) == 1:
            return [
                ipu.pipelining_ops.PipelineStageOptions(
                    matmul_options=get_matmul_options(available_memory_proportion[0], matmul_partials_type)
                )
                for stage in device_mapping
            ]
        else:
            raise ValueError(
                "Available memory proportion must be set for each of"
                f" the {num_ipus_per_replica} IPUs in the pipeline."
            )
    return [
        ipu.pipelining_ops.PipelineStageOptions(
            matmul_options=get_matmul_options(available_memory_proportion[stage], matmul_partials_type)
        )
        for stage in device_mapping
    ]


def pipeline_model(
    model, config, pipeline_names, pipeline_allocate_previous, num_pipeline_stages, matmul_partials_type
):
    """
    Returns the provided model with a new pipeline stage assignment.
    :param config: A config namespace object that contains details about
        the dataset split configuration, see `utilities/options.py`.
    :param pipeline_names: Dictionary of user friendly names matching
        layer identifiers, which can be the layer name or its class.
    :param pipeline_allocate_previous: A list of layers classes that
        will be assigned to the same pipeline stage as the previous layer.
    :param num_pipeline_stages: An int representing the number of IPUs
        used in a pipeline.
    :param pipelining_mode: A str defining the type pf pipelining mode.
    :return: The model pipelined.
    """
    pipeline_stages = config.ipu_opts.pipeline_stages
    # Set pipeline stages by assigning the specific layers to specific IPU decided by user
    # if the pipeline_stages is set properly
    if pipeline_stages is not None and len(pipeline_stages) > 1:
        logging.info("Using user specified pipelining strategy")
        pipeline_assigner = PipelineStagesAssigner(pipeline_allocate_previous, pipeline_names)
        assignments = model.get_pipeline_stage_assignment()
        assignments = pipeline_assigner.assign_pipeline_stages(assignments, pipeline_stages)
        device_mapping = config.ipu_opts.pipeline_device_mapping
    # Equal split layers to IPUs if pipeline_stages is not set
    else:
        logging.info("Using equal split pipelining strategy")
        assignments = model.get_pipeline_stage_assignment()
        device_mapping = get_pipeline_stages_equal_split(assignments, num_pipeline_stages, pipeline_allocate_previous)

    model.set_pipeline_stage_assignment(assignments)
    pipeline_schedule = get_poplar_options_per_pipeline_stage(
        num_pipeline_stages, device_mapping, config.ipu_opts.available_memory_proportion, matmul_partials_type
    )

    model.set_pipelining_options(
        gradient_accumulation_steps_per_replica=config.ipu_opts.gradient_accumulation_factor,
        device_mapping=device_mapping,
        forward_propagation_stages_poplar_options=pipeline_schedule,
        backward_propagation_stages_poplar_options=pipeline_schedule,
        recomputation_mode=ipu.pipelining_ops.RecomputationMode.RecomputeAndBackpropagateInterleaved,
        gradient_accumulation_reduction_method=ipu.optimizers.GradientAccumulationReductionMethod.RUNNING_MEAN,
        gradient_accumulation_dtype=str_dtype_to_tf_dtype(
            config.ipu_opts.gradient_accumulation_dtype or config.model.dtype
        ),
        offload_weight_update_variables=config.ipu_opts.offload_optimizer_state,
        replicated_optimizer_state_sharding=config.ipu_opts.RTS,
    )
    model.print_pipeline_stage_assignment_summary()
