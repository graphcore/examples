#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/tensorindex.hpp>
#include <popart/op.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/dropout.hpp>
#include <popart/popx/op/matmulx.hpp>
#include <popart/logging.hpp>
#include <queue>

template <class T>
static popart::Op *search_producers_for(popart::Tensor *t, int max_depth=-1) {
    // Searched as far as we can without success
    if (t->tensorType() == popart::TensorType::Variable || !t->hasProducer()) {
        return nullptr;
    }
    auto op = t->getProducer();
    if (op->isConvertibleTo<T>()) {
        return op;
    }

    if (op->input->n() < 1) {
        return nullptr;
    }

    if (op->input->n() > 1) {
        // TODO: Have whitelist of traversable ops
        if (!op->isConvertibleTo<popart::DropoutGradOp>()) {
            return nullptr;
        }
    }

    // Providing a max-search depth of -1 will remove the depth limit at the cost of potentially
    // unnecessary checks.
    if (max_depth > 0) {
        max_depth -= 1;
        if (max_depth == 0) {
            return nullptr;
        }
    }

    return search_producers_for<T>(op->input->tensors().front(), max_depth);
}

// Finds the underlying variable by searching through producers.
static popart::Tensor *get_variable(popart::Tensor *t) {
    if (t->tensorType() == popart::TensorType::Variable) {
        return t;
    } else if (!t->hasProducer()) {
        return nullptr;
    }
    auto op = t->getProducer();
    if (op->input->n() != 1) {
        return nullptr;
    }
    return get_variable(op->input->tensors().front());
}

// Attempts to find T by searching through consumers.
template <class T>
static popart::Op *search_consumers_for(popart::Tensor *w, std::queue<popart::Tensor *> &q) {
    for (auto consumer : w->consumers.getOps()) {
        if (consumer->isConvertibleTo<T>()) {
            return consumer;
        }

        // TODO: Have whitelist of traversable ops.
        if (consumer->isConvertibleTo<DetachOp>() || consumer->isConvertibleTo<popart::DropoutGradOp>()) {
            q.push(consumer->output->tensor(0));
        }
    }
    if (q.size() < 1) {
        return nullptr;
    }
    w = q.front();
    q.pop();
    return search_consumers_for<T>(w, q);
}
template <class T>
static popart::Op *search_consumers_for(popart::Tensor *w) {
    std::queue<popart::Tensor *> q;
    return search_consumers_for<T>(w, q);
}

template <class T>
static bool weight_consumed_by(popart::Tensor *w) {
    w = get_variable(w);
    if (w) {
        return search_consumers_for<T>(w) != nullptr;
    }
    return false;
}