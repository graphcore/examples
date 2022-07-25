// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once
#include <cstdint>

struct IRndGenerator {
    virtual ~IRndGenerator() = default;

    virtual void setRandomSeed(uint64_t seed) = 0;
    virtual uint32_t getRandom() = 0;
    virtual int32_t getRandom(int32_t from, int32_t to) = 0;
};
