// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_ROTARYPOSEMBED_OPIDS
#define GUARD_ROTARYPOSEMBED_OPIDS

#include <popart/attributes.hpp>
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/operatoridentifier.hpp>

using InMapType = std::map<popart::InIndex, popart::TensorId>;
using OutMapType = std::map<popart::OutIndex, popart::TensorId>;
using OutIndex = int;

namespace popart {

#define CUSTOM_OP_DOMAIN "popxl.addons.ops"

const popart::OperatorIdentifier RotaryPosEmbed = OperatorIdentifier{
    CUSTOM_OP_DOMAIN,
    "RotaryPosEmbed",
    1,      // Op version
    {3, 3}, // number of inputs
    1       // number of outputs
};

const popart::OperatorIdentifier RotaryPosEmbedGrad = OperatorIdentifier{
    CUSTOM_OP_DOMAIN,
    "RotaryPosEmbedGrad",
    1,      // Op version
    {3, 3}, // number of inputs
    1       // number of outputs
};

} // namespace popart

#endif
