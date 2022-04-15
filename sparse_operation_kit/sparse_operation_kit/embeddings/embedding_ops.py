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
import tensorflow.distribute as tf_dist
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
import sys, os

CommToolSet = set(["Strategy", "MPI", "Horovod", "OneDevice"])

def get_global_replica_id(comm_tool=None, var=None):
    def _strategy():
        def _get_current_replica_id_in_group_sync():
            replica_ctx = tf_dist.get_replica_context()
            if replica_ctx:
                replica_id = replica_ctx.replica_id_in_sync_group
            else:
                replica_id = (distribute_lib.get_update_replica_id() if kit_lib.in_tensorflow2() 
                             else distribute_lib.get_update_device())
            if replica_id is None:
                replica_id = array_ops.constant(0, dtype=array_ops.dtypes.int32)
            return replica_id

        return _get_current_replica_id_in_group_sync()

    def _MPI():
        return int(os.getenv("OMPI_COMM_WORLD_RANK"))

    def _Horovod():
        import horovod.tensorflow as hvd
        return hvd.local_rank()

    def _OneDevice():
        return 0

    if comm_tool is None:
        raise RuntimeError("SparseOperationKit can only works with "
                        "tf.distribute.Strategy, MPI, Horovod or single GPU.")
    
    if comm_tool not in CommToolSet:
        raise RuntimeError("SparseOperationKit only works with tf.distribute.Strategy, "
                        "MPI, Horovod or single GPU. But got %s" %comm_tool)

    if "Strategy" == comm_tool:
        return _strategy()
    elif "MPI" == comm_tool:
        return _MPI()
    elif "Horovod" == comm_tool:
        return _Horovod()
    elif "OneDevice" == comm_tool:
        return _OneDevice()

def _get_comm_tool():
    if "horovod.tensorflow" in sys.modules:
        return "Horovod"
    elif tf_dist.has_strategy():
        return "Strategy"
    elif os.getenv("OMPI_COMM_WORLD_SIZE") is not None:
        return "MPI"
    else:
        return "OneDevice"

def _get_embedding_variable_attr(embedding_variable, attr):
    if not isinstance(attr, str):
        raise ValueError("attr must be a string, but got {}".format(type(attr)))
    if hasattr(embedding_variable, "values"):
        return getattr(embedding_variable.values[0], attr)
    else:
        return getattr(embedding_variable, attr)

def embedding_lookup(embedding_variable, values, training=True, dynamic_input=False):
    """
    This function is a wrapper of SOK's dense forward propagation.
    """
    embedding_layer = embedding_variable.embedding_layer

    resource_variable_ops.variable_accessed(embedding_variable)

    global_replica_id = get_global_replica_id(_get_comm_tool(), embedding_variable)

    vector, _  = kit_lib.plugin_dense_fprop(embedding_variable._handle,
                                            embedding_layer.handle, 
                                            values=values,
                                            global_replica_id=global_replica_id,
                                            training=training,
                                            unique_op_name=embedding_variable.name,
                                            dynamic_input=dynamic_input,
                                            dtype=embedding_layer.compute_dtype)
    return vector


def embedding_lookup_sparse(embedding_variable, sp_ids, slot_num, training=True):
    """
    This function is a wrapper of SOK's sparse forward propagation.
    """
    if not isinstance(sp_ids, sparse_tensor.SparseTensor):
        raise TypeError("sp_ids must be SparseTensor")

    values = sp_ids.values
    indices = check_ops.ensure_shape(sp_ids.indices, shape=(None, 2))
    row_indices = array_ops.transpose(indices, perm=[1, 0])[0]

    embedding_layer = embedding_variable.embedding_layer

    resource_variable_ops.variable_accessed(embedding_variable)

    global_replica_id = get_global_replica_id(_get_comm_tool(), embedding_variable)

    vector, _ = kit_lib.plugin_sparse_fprop(embedding_variable._handle,
                                       embedding_layer.handle,
                                       values, row_indices,
                                       global_replica_id,
                                       slot_num=slot_num,
                                       training=training,
                                       unique_op_name=embedding_variable.name,
                                       dtype=embedding_layer.compute_dtype)
    return vector
