// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#define COMPAT_SPDLOG_VERSION (SPDLOG_VER_MAJOR * 10000 + SPDLOG_VER_MINOR * 100 + SPDLOG_VER_PATCH)

#include <spdlog/spdlog.h>
#if COMPAT_SPDLOG_VERSION >= 10500
#include <spdlog/sinks/stdout_sinks.h>
#endif
#include <spdlog/fmt/ostr.h>