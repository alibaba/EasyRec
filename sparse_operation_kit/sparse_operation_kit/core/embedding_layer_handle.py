"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sparse_operation_kit import kit_lib
from sparse_operation_kit.core.graph_keys import GraphKeys
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops

class EmbeddingLayerHandle(trackable.Trackable):
    """
    This is the base class used to track embedding layer handle.
    """
    def __init__(self,
                 embedding_variable,
                 compute_dtype,
                 **unused):
        super(EmbeddingLayerHandle, self).__init__()

        self._embedding_variable = embedding_variable
        self._compute_dtype = compute_dtype

        if hasattr(self._embedding_variable, "values"):
            for variable in self._embedding_variable.values:
                variable.set_embedding_layer(self)
        else:
            self._embedding_variable.set_embedding_layer(self)
    @property
    def initializer(self):
        return self._initializer_op

    @property
    def handle(self):
        return self._handle

    @property
    def compute_dtype(self):
        return self._compute_dtype

    def __repr__(self):
        return (f"<sok.EmbeddingLayerHandle '{type(self).__name__}' "
                f"pointed to {self._embedding_variable.name}>")

class DenseEmbeddingLayerHandle(EmbeddingLayerHandle):
    """
    This is the handle for dense embedding layer,
    which means no reduction conducted intra slots.
    """
    def __init__(self,
                 embedding_variable,
                 input_dispatcher,
                 embedding_lookuper,
                 output_dispatcher,
                 input_dispatcher_subsequent_ops = [],
                 output_dispatcher_subsequent_ops = [],
                 slot_num = 1,
                 nnz_per_slot = 1,
                 compute_dtype=None,
                 **unused):
        super(DenseEmbeddingLayerHandle, self).__init__(embedding_variable, compute_dtype)

        self._embedding_variable = embedding_variable
        self._input_dispatcher = input_dispatcher
        self._input_dispatcher_subsequent_ops = input_dispatcher_subsequent_ops
        self._embedding_lookuper = embedding_lookuper
        self._output_dispatcher = output_dispatcher
        self._output_dispatcher_subsequent_ops = output_dispatcher_subsequent_ops
        self._slot_num = slot_num
        self._nnz_per_slot = nnz_per_slot

        with ops.init_scope():

            if hasattr(self._embedding_variable, "values"):
                emb_var_handle = self._embedding_variable.values[0].emb_handle
                emb_var_name = self._embedding_variable.values[0].m_var_name
            else:
                emb_var_handle = self._embedding_variable.emb_handle
                emb_var_name = self._embedding_variable.m_var_name

            self._handle = kit_lib.create_embedding_dense(emb_var_handle,
                                            input_dispatcher=self._input_dispatcher,
                                            input_dispatcher_subsequent_ops=self._input_dispatcher_subsequent_ops,
                                            embedding_lookuper=self._embedding_lookuper,
                                            output_dispatcher=self._output_dispatcher,
                                            output_dispatcher_subsequent_ops=self._output_dispatcher_subsequent_ops,
                                            slot_num=self._slot_num,
                                            nnz_per_slot=self._nnz_per_slot,
                                            layer_handle_name=emb_var_name,
                                            compute_dtype=self.compute_dtype)

            self._initializer_op = control_flow_ops.group((self._handle))

            collections = [GraphKeys.SparseOperationKitEmbeddingLayers]
            ops.add_to_collections(collections, self)

class SparseEmbeddingLayerHandle(EmbeddingLayerHandle):
    """
    This is the handle for sparse embedding layer.
    which means reduction will be conducted intra slots.
    """
    def __init__(self,
                 embedding_variable,
                 input_dispatcher,
                 embedding_executor,
                 output_dispatcher,
                 input_dispatcher_subsequent_ops = [],
                 output_dispatcher_subsequent_ops = [],
                 slot_num = 1,
                 max_nnz = 1,
                 max_feature_num = 1,
                 combiner="sum",
                 compute_dtype=None):
        super(SparseEmbeddingLayerHandle, self).__init__(embedding_variable, compute_dtype)

        self._embedding_variable = embedding_variable
        self._input_dispatcher = input_dispatcher
        self._embedding_executor = embedding_executor
        self._output_dispatcher = output_dispatcher
        self._input_dispatcher_subsequent_ops = input_dispatcher_subsequent_ops
        self._output_dispatcher_subsequent_ops = output_dispatcher_subsequent_ops
        self._slot_num = slot_num
        self._max_nnz = max_nnz
        self._max_feature_num = max_feature_num
        self._combiner = combiner

        with ops.init_scope():
            if hasattr(self._embedding_variable, "values"):
                emb_var_handle = self._embedding_variable.values[0].emb_handle
                emb_var_name = self._embedding_variable.values[0].m_var_name
            else:
                emb_var_handle = self._embedding_variable.emb_handle
                emb_var_name = self._embedding_variable.m_var_name

            self._handle = kit_lib.create_embedding_sparse(emb_var_handle,
                                                input_dispatcher=self._input_dispatcher,
                                                input_dispatcher_subsequent_ops=self._input_dispatcher_subsequent_ops,
                                                embedding_executor=self._embedding_executor,
                                                output_dispatcher=self._output_dispatcher,
                                                output_dispatcher_subsequent_ops=self._output_dispatcher_subsequent_ops,
                                                slot_num=self._slot_num,
                                                max_nnz=self._max_nnz,
                                                max_feature_num=self._max_feature_num,
                                                combiner=self._combiner,
                                                layer_handle_name=emb_var_name,
                                                compute_dtype=self.compute_dtype)

            self._initializer_op = control_flow_ops.group((self._handle))

            collections = [GraphKeys.SparseOperationKitEmbeddingLayers]
            ops.add_to_collections(collections, self)
