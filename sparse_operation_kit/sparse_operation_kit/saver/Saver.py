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
from sparse_operation_kit import kit_lib
from sparse_operation_kit.core.embedding_layer_handle import GraphKeys
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

# TODO: make it inherit from trackable???
class Saver(object):
    """
    Abbreviated as ``sok.Saver()``.
    """
    def __init__(self):
        # TODO: how to get all emb_var from Model???
        pass

    def __call__(self):
        pass

    def dump_to_file(self, embedding_variable, filepath):
        """
        This function is used to save the specified embedding variables 
        to host file.

        When multiple CPU processes is used, this function must be called 
        within each CPU processes.

        Parameters
        ----------
        embedding_variable: sok.EmbeddingVariable, tf.DistributedVariable
                The variable from embedding layer which needs to be dumped
                to file.
        filepath: string
                The directory where the parameters will be dumped to.
        
        Returns
        -------
        status: tf.Tensor
                If this op executed successfully, then 'OK' will be returned.
        """
        # TODO: check whether embedding_variable is an instance of DistributedVariable
        if hasattr(embedding_variable, "emb_handle"):
            # horovod branch
            return kit_lib.dump_to_file(embedding_variable.emb_handle, filepath)
        else:
            # strategy branch
            return kit_lib.dump_to_file(embedding_variable.values[0].emb_handle, filepath)

    def restore_from_file(self, embedding_variable, filepath):
        """
        This function is used to restore dumped parameters to the specified embedding variable.
        
        When multiple CPU processes is used, this function must be called 
        within each CPU processes.

        Parameters
        ----------
        embedding_variable: sok.EmbeddingVariable, tf.DistributedVariable
                The embedding variable which needs to be restored from file.
        filepath: string
                The directory where the parameters will be restored from.

        Returns
        -------
        status: tf.Tensor
                If this op executed successfully, then 'OK' will be returned.
        """
        if kit_lib.in_tensorflow2():
            context = ops.NullContextmanager
            initializers = None
        else:
            context = ops.control_dependencies
            # in case the embedding layer has not been created
            collections = ops.get_collection(GraphKeys.SparseOperationKitEmbeddingLayers)
            initializers = [collect.initializer for collect in collections]

        with context(initializers):
            if hasattr(embedding_variable, "emb_handle"):
                # horovod branch
                return kit_lib.restore_from_file(embedding_variable.emb_handle, filepath)
            else:
                # strategy branch
                return kit_lib.restore_from_file(embedding_variable.values[0].emb_handle, filepath)

    def load_embedding_values(self, embedding_variable, tensors):
        """
        This function is used to assign embedding_variable's value with tf.Tensors.

        When multiple CPU processes is used, this function must be called 
        within each CPU processes.

        Parameters
        ----------
        embedding_variable: sok.EmbeddingVariable, tf.DistributedVariable
                    Which embedding_variable's value will be assigned.
        tensors: tf.Tensor, list of tf.Tensor, tuple of tf.Tensor
                    Each tf.Tensor must be 2-rank and the shape must be `[None, embedding_vec_size]`,
                    where the `embedding_vec_size` must be equal to that of embedding_variable's. 
                    All tf.Tensors make up to a big tensor, which just like they are stacked. For example:
                    `[tf.Tensor(shape=(bs_0, embedding_vec_size)), tf.Tensor(shape=(bs_1, embedding_vec_size)),\
                      tf.Tensor(shape=(bs_2, embedding_vec_size))]` will be treated as 
                    `tf.Tensor(shape=(bs_0 + bs_1 + bs_2, embedding_vec_size))`.

        Returns
        -------
        status: tf.Tensor
                If this op executed successfully, then 'OK' will be returned.
        """
        if kit_lib.in_tensorflow2():
            context = ops.NullContextmanager
            initializers = None
        else:
            context = ops.control_dependencies
            # in case the embedding layer has not been created
            collections = ops.get_collection(GraphKeys.SparseOperationKitEmbeddingLayers)
            initializers = [collect.initializer for collect in collections]

        if isinstance(tensors, list) or isinstance(tensors, tuple):
            # stack those tensors along dim-0
            tensors = array_ops.concat(tensors, axis=0)

        with context(initializers):
            if hasattr(embedding_variable, "emb_handle"):
                # horovod branch
                return kit_lib.load_embedding_values(embedding_variable.emb_handle, tensors)
            else:
                # strategy branch
                return kit_lib.load_embedding_values(embedding_variable.values[0].emb_handle, tensors)
