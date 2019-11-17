# Copyright 2019 Graphcore Ltd.
import platform
import socket

_getters = {}


def register_host_info(function):
    name = function.__name__
    if name.startswith('_'):
        name = name[1:]
    _getters[name] = function


def get_host_info():
    return {
        name: getter() for name, getter in _getters.items()
    }


@register_host_info
def _network_name():
    return platform.node()


@register_host_info
def _hostname():
    return socket.gethostname()


@register_host_info
def _machine():
    return platform.machine()


@register_host_info
def _python_version():
    return platform.python_version()


@register_host_info
def _machinable_version():
    # todo: dynamic encoding that aligns with the setup.cfg
    return '1.0.0rc'
