# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di
import yaml
import os
from config import cfg


def merge_dict(src, dst):
    for key, val in src.items():
        if isinstance(val, dict):
            if key not in dst:
                dst[key] = {}
            merge_dict(val, dst[key])
        else:
            dst[key] = val
    return dst


def assgin_attr(yaml_cfg, dst_cfg):
    for key, val in yaml_cfg.items():
        if isinstance(val, dict):
            assgin_attr(val, dst_cfg.get(key))
        else:
            dst_cfg.__setattr__(key, val)


def change_cfg_by_yaml_file(yaml_file):
    with open(yaml_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    yaml_cfg = load_parent(yaml_cfg)
    if 'parent' in yaml_cfg:
        del yaml_cfg['parent']
    if yaml_cfg is None:
        yaml_cfg = {}
    _, filename = os.path.split(yaml_file)
    name, _ = filename.split('.')
    assert 'task_name' not in yaml_cfg
    yaml_cfg['task_name'] = name
    assgin_attr(yaml_cfg, cfg)


def load_parent(yaml_cfg):
    if 'parent' in yaml_cfg:
        assert os.path.exists(yaml_cfg['parent'])
        with open(yaml_cfg['parent'], 'r') as f:
            parent_cfg = yaml.load(f, Loader=yaml.FullLoader)
        parent_cfg = load_parent(parent_cfg)
        yaml_cfg = merge_dict(yaml_cfg, parent_cfg)

    return yaml_cfg


def save_yaml(file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)


if __name__ == '__main__':
    change_cfg_by_yaml_file('yamls/example.yaml')
    print(cfg.TRAIN.STEPSIZE)
    print(cfg.TRAIN.LEARNING_RATE)
    print(cfg.TRAIN.BBOX_NORMALIZE_STDS)
    print(cfg['TRAIN']['BBOX_NORMALIZE_STDS'])
    save_yaml('test.yaml')
