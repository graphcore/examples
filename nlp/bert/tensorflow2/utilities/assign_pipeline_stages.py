# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from collections import defaultdict
import inspect


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
        return assignments

    def check_valid_model_expansion(self, assignments):
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
        if layer.name in self.pipeline_allocate_previous or type(layer) in self.pipeline_allocate_previous:
            return stage_id

        for i, stage_slots in enumerate(num_layers_per_stage):
            valid_elements = [i for k in stage_slots.keys() for i in self.pipeline_names[k]]
            valid_names = tuple(name for name in valid_elements if isinstance(name, str))
            valid_types = tuple(name for name in valid_elements if inspect.isclass(name))
            if layer.name in valid_names or (isinstance(layer, valid_types) and pipeline_index[type(layer)] != "hid"):
                return i
            if isinstance(layer, valid_types) and pipeline_index[type(layer)] == "hid" and stage_slots["hid"] > 0:
                num_layers_per_stage[i]["hid"] -= 1
                return i
        raise Exception(
            f"No available slot for layer `{layer.name}` of type `{type(layer)}` " f"with given pipeline stages."
        )


class GluePipelineStagesAssigner(PipelineStagesAssigner):
    def assign_glue_pipeline_stages(self, assignments, pipeline_stages):
        dropout_layer_name = assignments[-2].layer.name
        if "dropout" not in dropout_layer_name:
            raise ValueError(
                "Error in allocating the TFBertForSequenceClassification "
                + "model to pipeline stages.In attempting to replace the name "
                + "of the final dropout layer, a name containing 'dropout' was "
                + "expected. Found instead {dropout_layer_name}."
            )
        self.pipeline_names["glue_head"].append(dropout_layer_name)
        return super().assign_pipeline_stages(assignments, pipeline_stages)
