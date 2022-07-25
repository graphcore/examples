// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <popart/ir.hpp>
#include <popart/graph.hpp>
#include <popart/op.hpp>
// #include <popart/opidentifier.hpp>
#include <popart/operators.hpp>
#include <popart/tensornames.hpp>
#include <popart/op/add.hpp>
#include <popart/op/dropout.hpp>
#include <popart/op/groupnorm.hpp>
#include <popart/op/reshape.hpp>

#include <popart/graphutils.hpp>


namespace {
bool produced_by_dropout(popart::Tensor *t) {
    return t->hasProducer() && t->getProducer()->isConvertibleTo<popart::DropoutBaseOp>();
}

bool has_priorites_set(popart::Op *op) {
    return !op->settings.inplacePriorityVeto.empty();
}
}

// This Pattern find the residual adds in GPT2 and encorages PopART
// to inplace on the output created by group_norm.
// This is beneficial as the Tensor layout of LHS in the first residual tensor
// comes from the embedding output which is less memory balanced.
class ResidualAddInPlacePattern : public popart::PreAliasPattern {
public:
    bool matches(popart::Op *op) const override {
        if (op->isConvertibleTo<popart::AddOp>()) {
            return produced_by_dropout(op->inTensor(popart::AddOp::getArg0InIndex()))
                    && !has_priorites_set(op);
        }
        return false;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        op->settings.inplacePriorityVeto.push_back({"AddLhsInplace", 1000.0f});
        return true;
    }
};


static popart::PatternCreator<ResidualAddInPlacePattern> c1("ResidualAddInPlacePattern", true);