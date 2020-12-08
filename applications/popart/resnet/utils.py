# Copyright (c) 2019 Graphcore Ltd. All rights reserved.


class NullContextManager(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
