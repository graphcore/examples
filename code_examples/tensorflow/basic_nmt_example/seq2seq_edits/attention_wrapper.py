# Copyright 2019 Graphcore Ltd.
"""
Edits to seq2seq AttentionWrapper
"""

import tensorflow as tf


class AttentionWrapperNoAssert(tf.contrib.seq2seq.AttentionWrapper):
    # Stops the adding of Assert operations that "assert_equal" the wrapper batch_size and the attention_mechanisms batch_size
    def _batch_size_checks(self, batch_size, error_message):
        return []
