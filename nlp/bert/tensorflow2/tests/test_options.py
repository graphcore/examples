# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from argparse import Namespace

from tests.utils import get_app_root_dir
from utilities.argparser import combine_config_file_with_args, merge_args_with_options
from utilities.options import GLUEOptions, PretrainingOptions, SQuADOptions


class TestConfigs:
    @classmethod
    def setup_class(cls):
        cls.app_dir = get_app_root_dir()
        cls.config_dir = cls.app_dir.joinpath("configs")
        cls.test_config_dir = cls.app_dir.joinpath("tests")

    @staticmethod
    def get_config_names(path, match):
        configs = path.glob(f"{match}_*.json")
        return [config for config in configs]

    def test_pretrain_configs_contain_required_options(self):
        match_str = "pretrain"
        pretrain_configs = self.get_config_names(self.config_dir, match_str)
        test_pretrain_configs = self.get_config_names(self.test_config_dir, match_str)
        assert len(pretrain_configs) > 0, f"Can't find any pretraining configs in directory {self.config_dir}"
        assert len(test_pretrain_configs) > 0, f"Can't find any pretraining configs in directory {self.test_config_dir}"

        for config in pretrain_configs + test_pretrain_configs:
            print(f"Testing config: {config}")
            combine_config_file_with_args(Namespace(config=config), PretrainingOptions)

    def test_squad_configs_contain_required_options(self):
        match_str = "squad"
        squad_configs = self.get_config_names(self.config_dir, match_str)
        test_squad_configs = self.get_config_names(self.test_config_dir, match_str)
        assert len(squad_configs) > 0, f"Can't find any squad configs in directory {self.config_dir}"
        assert len(test_squad_configs) > 0, f"Can't find any squad configs in directory {self.test_config_dir}"

        for config in squad_configs + test_squad_configs:
            print(f"Testing config: {config}")
            combine_config_file_with_args(Namespace(config=config), SQuADOptions)

    def test_glue_configs_contain_required_options(self):
        match_str = "glue"
        glue_configs = self.get_config_names(self.config_dir, match_str)
        test_glue_configs = self.get_config_names(self.test_config_dir, match_str)
        assert len(glue_configs) > 0, f"Can't find any GLUE configs in directory {self.config_dir}"
        assert len(test_glue_configs) > 0, f"Can't find any GLUE configs in directory {self.test_config_dir}"

        for config in glue_configs + test_glue_configs:
            print(f"Testing config: {config}")
            combine_config_file_with_args(Namespace(config=config), GLUEOptions)


def test_merge_args_with_options():
    args = {"a.b.c": 5, "d": 3, "e.f": "."}
    config = {"a": {"b": {"c": 6}}, "d": 4, "e": {"g": 2}, "h": 3}
    expected_merged_options = {"a": {"b": {"c": 5}}, "d": 3, "e": {"g": 2, "f": "."}, "h": 3}
    merged_options = merge_args_with_options(args, config)
    assert expected_merged_options == merged_options
