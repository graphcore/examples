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


def get_multi_level_folders(path):
    _folder, _file = os.path.split(path)
    if _folder in ['.', '/', '', '..']:
        return [_folder, _file]
    _folders = get_multi_level_folders(_folder)
    return _folders + [_file]


def change_cfg_by_yaml_file(yaml_file):
    with open(yaml_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    yaml_cfg = load_parent(yaml_cfg)
    if 'parent' in yaml_cfg:
        del yaml_cfg['parent']
    if yaml_cfg is None:
        yaml_cfg = {}

    # assign yaml file name to name of this config
    folderpath, filename = os.path.split(yaml_file)
    name, _ = filename.split('.')
    assert 'task_name' not in yaml_cfg
    yaml_cfg['task_name'] = name

    # assign output dir, if yaml file in yamls folder, replace folder 'yamls' with 'outputs', then the result path is the output dir
    # else use 'output' + name as the output dir
    list_of_folders_and_file = get_multi_level_folders(folderpath)
    if 'yamls' in list_of_folders_and_file:
        idx = list_of_folders_and_file.index('yamls')
        list_of_folders_and_file[idx] = 'outputs'
        list_of_folders_and_file = list_of_folders_and_file + [name]
        output_dir = os.path.join(*list_of_folders_and_file)
    else:
        output_dir = os.path.join('outputs', name)
    yaml_cfg['output_dir'] = output_dir

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
