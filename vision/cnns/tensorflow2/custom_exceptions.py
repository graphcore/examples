# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


class UnsupportedFormat(TypeError):
    pass


class DimensionError(ValueError):
    pass


class MissingArgumentException(ValueError):
    pass


class InvalidPrecisionException(NameError):
    pass


class UnallowedConfigurationError(ValueError):
    pass
