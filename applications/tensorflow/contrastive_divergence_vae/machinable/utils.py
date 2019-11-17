# Copyright 2019 Graphcore Ltd.
from collections import Mapping
import json


class tcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def serialize(obj):
    """JSON serializer for objects not serializable by default json code
    """
    if getattr(obj, '__dict__', False):
        return obj.__dict__

    return str(obj)


def msg(text, level='info', color=None):
    if isinstance(color, str) and hasattr(tcolors, color.upper()):
        text = getattr(tcolors, color.upper()) + text + tcolors.ENDC

    print(text)


def prettydict(dict_like, sort_keys=False):
    try:
        return json.dumps(dict_like, indent=4, default=serialize, sort_keys=sort_keys)
    except TypeError:
        return str(dict_like)


def update_dict(d, update=None, copy=False):
    if d is None:
        d = {}
    if copy:
        d = d.copy()
    if not update or update is None:
        return d
    for k, v in update.items():
        if isinstance(v, Mapping):
            d[k] = update_dict(d.get(k, {}), v, copy=copy)
        else:
            d[k] = v
    return d
