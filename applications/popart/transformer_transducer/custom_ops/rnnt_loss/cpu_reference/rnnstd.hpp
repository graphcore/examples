// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once

void cost_and_grad(float *log_probs, float *grads, float *costs,
                   int *flat_labels, int *label_lengths, int *input_lengths,
                   int batch_size, int max_t, int max_u, int alphabet_size,
                   int blank);

