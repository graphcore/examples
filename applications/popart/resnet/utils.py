# Copyright 2019 Graphcore Ltd.
class NullContextManager(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
