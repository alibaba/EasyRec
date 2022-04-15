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
from sparse_operation_kit.core.inplace_initializer import InPlaceInitializer
from tensorflow.python.keras import initializers as tf_initializers
from tensorflow.python.ops.resource_variable_ops import BaseResourceVariable, variable_accessed, _maybe_set_handle_data
from tensorflow.python.ops.resource_variable_ops import _handle_graph
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework import dtypes
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.distribute import get_strategy, has_strategy
from tensorflow.python.ops.variables import VariableSynchronization, VariableAggregation
from tensorflow.python.distribute.values import DistributedVariable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import control_flow_ops
import functools

class EmbeddingVariable(BaseResourceVariable):
    """
    EmbeddingVariable used in TF1.x
    """
    @classmethod
    def CreateInstances(cls, *args, **kwargs):
        if not has_strategy():
            return EmbeddingVariable(local_replica_id=0, *args, **kwargs)

        strategy = get_strategy()
        strategy_extended = strategy.extended
        devices = strategy_extended._devices
        
        value_list = []
        for i, d in enumerate(devices):
            with ops.device(d):
                if i > 0:
                    name = value_list[0].name.split(":")[0]
                    kwargs["name"] = "%s/replica_%d/" % (name, i)
                with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
                    with tape.stop_recording():
                        v = EmbeddingVariable(local_replica_id=i, *args, **kwargs)
                value_list.append(v)
        
        # TODO: check whether it will impact the performance due to the aggregation or synchronization setting.
        return DistributedVariable(strategy=strategy, values=value_list,
                                    aggregation=VariableAggregation.ONLY_FIRST_REPLICA,
                                    var_policy=VariableSynchronization.NONE)

    def __init__(self,
                 shape,
                 local_replica_id,
                 initializer=None,
                 trainable=True,
                 use_hashtable=True,
                 name="EmbeddingVariable",
                 dtype=None,
                 key_dtype=None,
                 *args,
                 **kwargs):
        if (not isinstance(shape, list)) or (len(shape) != 2):
            raise ValueError("shape_per_gpu must be a list which represents: "+\
                             "[vocabulary_size_per_gpu, embedding_vector_size].")
        self.m_shape_per_gpu = TensorShape(shape)
        self.m_local_replica_id = local_replica_id
        self.m_initializer = initializer or InPlaceInitializer(name="random_uniform")
        if not isinstance(self.m_initializer, InPlaceInitializer):
            self.m_initializer = tf_initializers.get(self.m_initializer)
        self.m_trainable = trainable
        self.m_use_hashtable = use_hashtable
        self.m_embedding_layer = None
        self.m_dtype = dtype or dtypes.float32
        self.m_key_dtype = key_dtype or dtypes.int64
        # produce intial_value
        if isinstance(self.m_initializer, InPlaceInitializer):
            # TODO: serialize it
            self.m_initial_value = self.m_initializer.name
        else:
            self.m_initial_value = self.m_initializer(shape=self.m_shape_per_gpu, dtype=self.m_dtype)

        collections = [ops.GraphKeys.GLOBAL_VARIABLES]
        if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
            collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]

        with ops.init_scope():
            self._in_graph_mode = not context.executing_eagerly()
            with ops.name_scope(name) as var_name_scope:
                # TODO: use regulare expression 
                while var_name_scope[-1] == r"/":
                    var_name_scope = var_name_scope[:-1]
                var_name = var_name_scope
                self.m_var_name = var_name
                self.m_unique_id = "%s_%d" %(var_name, ops.uid())

                # attr = resource_variable_ops.attr_value_pb2.AttrValue(
                #     list=resource_variable_ops.attr_value_pb2.AttrValue.ListValue(
                #         s=[resource_variable_ops.compat.as_bytes("loc:@%s" % self.m_var_name)]))

                # with ops.get_default_graph()._attr_scope({"_class": attr}):
                with ops.NullContextmanager():
                    # m_handle is the handle to EmbeddingVariable, tf_handle is the handle to TF Var.
                    self.m_handle, self.tf_handle = kit_lib.create_var(var_name=var_name,
                                                               dtype=self.m_dtype,
                                                               shape=self.m_shape_per_gpu)

                    if self._in_graph_mode:
                        with ops.name_scope("IsInitialized"):
                            self._is_initialized_op = ops.convert_to_tensor(True) # TODO: should not hard-writing???

                            if (isinstance(self.m_initial_value, ops.Tensor) and 
                                not self.m_initial_value.shape.is_compatible_with(self.m_shape_per_gpu)):
                                raise ValueError("The initial value's shape (%s) is not compatible with "
                                                 "the explicitly supplied `shape` argument (%s)." %
                                                 (initial_value.shape, self.m_shape_per_gpu))

                            _init_op = kit_lib.assign_embedding_variable(emb_var_handle=self.m_handle,
                                                                 tf_var_handle=self.tf_handle,
                                                                 var_name=var_name,
                                                                 initial_value=self.m_initial_value,
                                                                 local_replica_id=self.m_local_replica_id,
                                                                 trainable=self.m_trainable,
                                                                 shape=self.m_shape_per_gpu,
                                                                 use_hashtable=self.m_use_hashtable,
                                                                 dtype=self.m_dtype,
                                                                 key_dtype=self.m_key_dtype)
                            self._initializer_op = control_flow_ops.group((_init_op))
                    else:
                        raise RuntimeError("Currently, EmbeddingVariable does not support Eager mode.")

                    if not context.executing_eagerly():
                        ops.add_to_collections(collections, self)

            super(EmbeddingVariable, self).__init__(trainable=self.m_trainable,
                                                    shape=self.m_shape_per_gpu,
                                                    dtype=self.m_dtype,
                                                    handle=self.m_handle,
                                                    handle_name=var_name,
                                                    distribute_strategy=get_strategy() if has_strategy() else None,
                                                    synchronization=VariableSynchronization.NONE,
                                                    aggregation=VariableAggregation.ONLY_FIRST_REPLICA,
                                                    unique_id=self.m_unique_id,
                                                    initializer_op=self._initializer_op,
                                                    is_initialized_op=self._is_initialized_op,
                                                    *args, **kwargs)
            handle_data = resource_variable_ops.cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
            handle_data.is_set = True
            handle_data.shape_and_type.append(
                resource_variable_ops.cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(
                    shape=self.shape.as_proto(), dtype=self.dtype.as_datatype_enum))
            resource_variable_ops._set_handle_shapes_and_types(self.m_handle, handle_data, 
                graph_mode=False if context.executing_eagerly() else True)
            resource_variable_ops._set_handle_shapes_and_types(self.tf_handle, handle_data, 
                graph_mode=False if context.executing_eagerly() else True)

    @property
    def emb_handle(self):
        return self.m_handle

    def set_embedding_layer(self, embedding_layer):
        if self.m_embedding_layer is not None:
            raise ValueError("EmbeddingLayer for %s is already set." %(self.name))
        self.m_embedding_layer = embedding_layer
    
    @property
    def embedding_layer(self):
        if self.m_embedding_layer is None:
            raise ValueError("EmbeddingLayer for %s is not set." %(self.name))
        return self.m_embedding_layer

    def _read_variable_op(self):
        variable_accessed(self)
        result = kit_lib.read_embedding_variable(self._handle, self.tf_handle, self._dtype, self.name)
        _maybe_set_handle_data(self._dtype, self._handle, result)

        if not context.executing_eagerly():
            tape.record_operation("ReadEmbeddingVariableOp", [result], [self._handle, self.tf_handle],
                                  lambda x: [x, None])
        return result

    def __deepcopy__(self, memo):
        if not context.executing_eagerly():
            raise NotImplementedError(
                "__deepcopy__() is only available when eager execution is enabled.")
        copied_variable = EmbeddingVariable(
            shape=self.shape,
            local_replica_id=self.m_local_replica_id,
            initial_value=self.m_initial_value,
            trainable=self._trainable,
            constraint=self._constraint,
            dtype=self._dtype,
            name=self._shared_name,
            distribute_strategy=self._distribute_strategy)
        memo[self._unique_id] = copied_variable
        return copied_variable

    def __reduce__(self):
        # The implementation mirrors that of __deepcopy__.
        return functools.partial(
            EmbeddingVariable,
            initial_value=self.m_initial_value,
            trainable=self.trainable,
            name=self._shared_name,
            dtype=self.dtype,
            constraint=self.constraint,
            distribute_strategy=self._distribute_strategy), ()

    def count_up_to(self, limit):
        raise NotImplementedError("EmbeddingVariable.count_up_to is not implemented.")

    def value(self):
        """A cached operation which reads the value of this variable."""
        if self._cached_value is not None:
            return self._cached_value
        with ops.colocate_with(None, ignore_existing=True):
            with ops.device(self._handle.device):
                return self._read_variable_op()

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        del name
        if dtype is not None and not dtype.is_compatible_with(self.dtype):
            raise ValueError(
                "Incompatible type conversion requested to type {!r} for variable "
                "of type {!r}".format(dtype.name, self.dtype.name))
        if as_ref:
            return self.read_value().op.inputs[0]
        else:
            return self.value()

    def _gather_saveables_for_checkpoint(self):
        """Overrides Trackable method.

        This allows both name-based and object-based save and restore of
        DistributedVariables.

        Returns:
        A dictionary mapping attribute names to `SaveableObject` factories.
        """
        raise NotImplementedError("_gather_saveables_for_checkpoint not implemented.")
        # def _saveable_factory(name=self._common_name):
        #     return _DistributedVariableSaveable(self, self._primary, name)

        # return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

    def sparse_read(self, indices, name=None):
        raise NotImplementedError("EmbeddingVariable.sparse_read is not implemented.")

    def gather_nd(self, indices, name=None):
        raise NotImplementedError("EmbeddingVariable.gather_nd is not implemented")

    def to_proto(self, export_scope=None):
        raise NotImplementedError("EmbeddingVariable.to_proto is not implemented")

    @staticmethod
    def from_proto(variable_def, import_scope=None):
        raise NotImplementedError("EmbeddingVariable.from_proto is not implemented.")

    @property
    def initializer(self):
        return self._initializer_op

    def is_initialized(self, name=None):
        raise NotImplementedError("EmbeddingVariable.is_initialized is not implemented")

    # # TODO: if TF optimizer need to be used, then leave it for TF implementations.
    # def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    #     return control_flow_ops.group((kit_lib.embedding_variable_assign_sub(self.m_handle, self.tf_handle, delta)))

    def assign_add(self, delta, use_locking=None, name=None, read_value=True):
        raise NotImplementedError("EmbeddingVariable.assign_add is not implemented.")

    # TODO: if TF optimizer need to be used, then leave it for TF implementations.
    # def _lazy_read(self, op):
        # raise NotImplementedError("EmbeddingVariable._lazy_read is not implemented.")

    def assign(self, value, use_locking=None, name=None, read_value=True):
        raise NotImplementedError("EmbeddingVariable.assign is not implemented.")

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        raise NotImplementedError("EmbeddingVariable.scatter_sub is not implemented")

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        raise NotImplementedError("EmbeddingVariable.scatter_add is not implemented")

    def scatter_max(self, sparse_delta, use_locking=False, name=None):
        raise NotImplementedError("EmbeddingVariable.scatter_max is not implemented.")

    def scatter_min(self, sparse_delta, use_locking=False, name=None):
        raise NotImplementedError("EmbeddingVariable.scatter_min is not implemented.")

    def scatter_mul(self, sparse_delta, use_locking=False, name=None):
        raise NotImplementedError("EmbeddingVariable.scatter_mul is not implemented.")

    def scatter_div(self, sparse_delta, use_locking=False, name=None):
        raise NotImplementedError("EmbeddingVariable.scatter_div is not implemented.")

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        raise NotImplementedError("EmbeddingVariable.scatter_update is not implemented.")

    def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
        raise NotImplementedError("EmbeddingVariable.batch_scatter_update is not implemented.")

    def scatter_nd_sub(self, indices, updates, name=None):
        raise NotImplementedError("EmbeddingVariable.scatter_nd_sub is not implemented.")

    def scatter_nd_add(self, indices, updates, name=None):
        raise NotImplementedError("EmbeddingVariable.scatter_nd_add is not implemented.")

    def scatter_nd_update(self, indices, updates, name=None):
        raise NotImplementedError("EmbeddingVariable.scatter_nd_update is not implemented.")

    def _strided_slice_assign(self, begin, end, strides, value, name, begin_mask,
                            end_mask, ellipsis_mask, new_axis_mask,
                            shrink_axis_mask):
        raise NotImplementedError("EmbeddingVariable._strided_slice_assign is not implemented.")

    def __int__(self):
        raise NotImplementedError("EmbeddingVariable.__int__ is not implemented.")

    def __iadd__(self, unused_other):
        raise NotImplementedError("EmbeddingVariable.__iadd__ is not implemented.")

    def __isub__(self, unused_other):
        raise NotImplementedError("EmbeddingVariable.__isub__ is not implemented.")

    def __imul__(self, unused_other):
        raise NotImplementedError("EmbeddingVariable.__imul__ is not implemented.")

    def __idiv__(self, unused_other):
        raise NotImplementedError("EmbeddingVariable.__idiv__ is not implemented.")

    def __itruediv__(self, unused_other):
        raise NotImplementedError("EmbeddingVariable.__itruediv__ is not implemented.")

    def __irealdiv__(self, unused_other):
        raise NotImplementedError("EmbeddingVariable.__irealdiv__ is not implemented.")

    def __ipow__(self, unused_other):
        raise NotImplementedError("EmbeddingVariable.__ipow__ is not implemented.")


    