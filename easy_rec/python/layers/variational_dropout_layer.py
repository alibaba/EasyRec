# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import tensorflow as tf

from easy_rec.python.compat.feature_column.feature_column import (
    _SharedEmbeddingColumn,  # NOQA
)
from easy_rec.python.compat.feature_column.feature_column_v2 import (
    EmbeddingColumn,  # NOQA
)

if tf.__version__ >= '2.0':
    tf = tf.compat.v1


class VariationalDropoutLayer(object):
    """Rank features by variational dropout.

    Use the Dropout concept on the input feature layer and optimize the corresponding feature-wise dropout rate
    paper: Dropout Feature Ranking for Deep Learning Models
    arXiv: 1712.08645
    """

    def __init__(self, variational_dropout_config, group_columns, is_training=False):
        self._config = variational_dropout_config
        self.features_dim_used = []
        self.features_embedding_size = 0
        for item in range(0, len(group_columns)):
            if hasattr(group_columns[item], 'dimension'):
                self.features_dim_used.append(group_columns[item].dimension)
                self.features_embedding_size += group_columns[item].dimension
            else:
                self.features_dim_used.append(1)
                self.features_embedding_size += 1

        if self.variational_dropout_wise():
            self._dropout_param_size = self.features_embedding_size
            self.drop_param_shape = [self._dropout_param_size]
        else:
            self._dropout_param_size = len(self.features_dim_used)
            self.drop_param_shape = [self._dropout_param_size]
        self.evaluate = not is_training

    def get_lambda(self):
        return self._config.regularization_lambda

    def variational_dropout_wise(self):
        return self._config.embedding_wise_variational_dropout

    def expand_bern_val(self):
        # Build index_list--->[[0,0],[0,0],[0,0],[0,0],[0,1]......]
        self.expanded_bern_val = []
        for i in range(len(self.features_dim_used)):
            index_loop_count = self.features_dim_used[i]
            for m in range(index_loop_count):
                self.expanded_bern_val.append([i])
        self.expanded_bern_val = tf.tile(self.expanded_bern_val, [self.batch_size, 1])
        batch_size_range = tf.range(self.batch_size)
        expand_range_axis = tf.expand_dims(batch_size_range, 1)
        self.fetures_dim_len = 0
        for i in self.features_dim_used:
            self.fetures_dim_len += self.features_dim_used[i]
        batch_size_range_expand_dim_len = tf.tile(expand_range_axis, [1, self.fetures_dim_len])
        index_i = tf.reshape(batch_size_range_expand_dim_len, [-1, 1])
        self.expanded_bern_val = tf.concat([index_i, self.expanded_bern_val], 1)

    def build_variational_dropout(self):
        self.logit_p = tf.get_variable(name='logit_p', shape=self.drop_param_shape, dtype=tf.float32, initializer=None)

    def sample_noisy_input(self, input):
        self.batch_size = tf.shape(input)[0]
        if self.evaluate:
            expanded_dims_logit_p = tf.expand_dims(self.logit_p, 0)
            expanded_logit_p = tf.tile(expanded_dims_logit_p, [self.batch_size, 1])
            p = tf.sigmoid(expanded_logit_p)
            if self.variational_dropout_wise():
                scaled_input = input * (1 - p)
            else:
                # expand dropout layer
                self.expand_bern_val()
                expanded_p = tf.gather_nd(p, self.expanded_bern_val)
                expanded_p = tf.reshape(expanded_p, [-1, self.fetures_dim_len])
                scaled_input = input * (1 - expanded_p)

            return scaled_input

        bern_val = self.sampled_from_logit_p(self.batch_size)
        bern_val = tf.reshape(bern_val, [-1, self.fetures_dim_len])
        noisy_input = input * bern_val
        return noisy_input

    def sampled_from_logit_p(self, num_samples):
        expand_dims_logit_p = tf.expand_dims(self.logit_p, 0)
        expand_logit_p = tf.tile(expand_dims_logit_p, [num_samples, 1])
        dropout_p = tf.sigmoid(expand_logit_p)
        bern_val = self.concrete_dropout_neuron(dropout_p)

        if self.variational_dropout_wise():
            return bern_val
        else:
            # from feature_num to embedding_dim_num
            self.expand_bern_val()
            bern_val_gather_nd = []
            bern_val_gather_nd = tf.gather_nd(bern_val, self.expanded_bern_val)
            return bern_val_gather_nd

    def concrete_dropout_neuron(self, dropout_p, temp=1.0 / 10.0):
        EPSILON = np.finfo(float).eps
        unif_noise = tf.random_uniform(tf.shape(dropout_p), dtype=tf.float32, seed=None, name='unif_noise')

        approx = (
            tf.log(dropout_p + EPSILON)
            - tf.log(1.0 - dropout_p + EPSILON)
            + tf.log(unif_noise + EPSILON)
            - tf.log(1.0 - unif_noise + EPSILON)
        )

        approx_output = tf.sigmoid(approx / temp)
        return 1 - approx_output

    def __call__(self, output_features):
        self.build_variational_dropout()
        noisy_input = self.sample_noisy_input(output_features)
        dropout_p = tf.sigmoid(self.logit_p)
        variational_dropout_penalty = 1.0 - dropout_p
        variational_dropout_penalty_lambda = self.get_lambda() / tf.cast(self.batch_size, dtype=tf.float32)
        variational_dropout_loss_sum = variational_dropout_penalty_lambda * tf.reduce_sum(
            variational_dropout_penalty, axis=0
        )
        tf.add_to_collection('variational_dropout_loss', variational_dropout_loss_sum)
        return noisy_input
