#
# Copyright (c) 2021, NVIDIA CORPORATION.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sparse_operation_kit import kit_lib
from tensorflow.python.ops import collective_ops
import tensorflow.distribute as tf_dist
MirroredStrategy = tf_dist.MirroredStrategy
try:   
    MultiWorkerMirroredStrategy = tf_dist.MultiWorkerMirroredStrategy
except AttributeError:
    MultiWorkerMirroredStrategy = tf_dist.experimental.MultiWorkerMirroredStrategy
from tensorflow import constant, TensorShape, function
from tensorflow.dtypes import int32, int64
from tensorflow import print as tf_print
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
import sys
from tensorflow.python.framework import config

def Init(**kwargs):
    """
    Abbreviated as ``sok.Init(**kwargs)``.

    This function is used to do the initialization of SparseOperationKit (SOK).

    SOK will leverage all available GPUs for current CPU process. Please set 
    `CUDA_VISIBLE_DEVICES` or `tf.config.set_visible_devices` to specify which 
    GPU(s) are used in this process before launching tensorflow runtime 
    and calling this function.

    In **TensorFlow 2.x**, SOK can be used with **tf.distribute.Strategy** or **Horovod**. 
    When it's used with tf.distribute.Strategy, it must be called under `strategy.scope()`. 
    For example,

    .. code-block:: python
    
        with strategy.scope():
            sok.Init(**kwargs)

    When it's used with Horovod, it must be called at each process. For example,

    .. code-block:: python
    
        import horovod.tensorflow as hvd

        hvd.init()

        sok.Init(**kwargs)

    In **TensorFlow 1.15**, SOK can only work with **Horovod**. The retured status
    must be evaluated with `sess.run`, and it must be the first step before evaluate
    any other SOK APIs.

    .. code-block:: python

        sok_init = sok.Init(global_batch_size=args.global_batch_size)
        with tf.Session() as sess:
            sess.run(sok_init)
            ...

    Parameters
    ----------
    kwargs: dictionary
            keyword arguments for this function. 
            Currently, it must contains `global_batch_size` used in all GPUs.

    Returns
    -------
    status: string
            a string will be returned if this function executed successfully.
            And its contents will be 'OK'.
    """

    def _get_visible_devices():
        gpus = config.get_visible_devices('GPU')
        assert(len(gpus) > 0)
        visible_devices = []
        for i in range(len(gpus)):
            visible_devices.append(int(gpus[i].name.split(':')[-1]))
        return array_ops.constant(visible_devices, dtype=int32)
    
    @function
    def _single_worker_init(**kwargs):
        replica_ctx = tf_dist.get_replica_context()
        replica_ctx.merge_call(lambda strategy: 
            tf_print("You are using the plugin with MirroredStrategy."))
        nccl_unique_id = replica_ctx.merge_call(lambda strategy:
                    kit_lib.get_nccl_unique_id())
        global_random_seed = kwargs.get("seed", None) or replica_ctx.merge_call(lambda strategy:
                                                                    kit_lib.gen_random_seed())

        global_id = replica_ctx.replica_id_in_sync_group
        visible_devices = _get_visible_devices()
        status = kit_lib.plugin_init(global_id, replica_ctx.num_replicas_in_sync, 
                                     nccl_unique_id, global_random_seed, visible_devices,
                                     global_batch_size=kwargs['global_batch_size']) 
        return status

    def _multi_worker_init(**kwargs):
        replica_ctx = tf_dist.get_replica_context()
        global_id = replica_ctx.replica_id_in_sync_group
        if global_id == 0:
            unique_id = kit_lib.get_nccl_unique_id()
            re = collective_ops.broadcast_send(unique_id,
                                                TensorShape([32,]),
                                                int32,
                                                group_size=replica_ctx.num_replicas_in_sync,
                                                group_key=1,
                                                instance_key=2)
        else:
            re = collective_ops.broadcast_recv(TensorShape([32,]),
                                                int32,
                                                group_size=replica_ctx.num_replicas_in_sync,
                                                group_key=1,
                                                instance_key=2)
        if global_id == 0:
            global_seed = kwargs.get("seed", None) or kit_lib.gen_random_seed()
            re_seed = collective_ops.broadcast_send(global_seed,
                                                TensorShape([1,]),
                                                int64,
                                                group_size=replica_ctx.num_replicas_in_sync,
                                                group_key=1,
                                                instance_key=3)
        else:
            global_seed = kwargs.get("seed", None)
            re_seed = collective_ops.broadcast_recv(TensorShape([1,]),
                                                int64,
                                                group_size=replica_ctx.num_replicas_in_sync,
                                                group_key=1,
                                                instance_key=3)

            if (global_seed and global_seed != re_seed):
                logging.warning("The seed: {} is not consistent with that from cheif-node: {}, "
                                "and the seed from cheif-node will be used.".format(global_seed, re_seed))

        visible_devices = _get_visible_devices()
        status = kit_lib.plugin_init(global_id, replica_ctx.num_replicas_in_sync, 
                                     re, re_seed, visible_devices,
                                     global_batch_size=kwargs['global_batch_size'])
        return status

    # @function
    def _horovod_init(**kwargs):
        r"""
        This function uses horovod to broadcast nccl-id and random-seed which is used by sparse_operation_kit.
        Please note that the nccl-comm mentioned here is not the same one as the nccl-comm of horovod itself.

        After broadcasting, this function uses kit_lib.plugin_init and "nccl-id", "random-seed" to initialize 
        sparse_operation_kit.
        """
        local_rank = hvd.local_rank()

        unique_id = kit_lib.get_nccl_unique_id() if local_rank == 0 else array_ops.zeros([32,], dtype=int32)
        unique_id = hvd.broadcast(unique_id, root_rank=0, name="nccl_unique_id")

        seed = kwargs.get("seed", None)
        if 0 == local_rank:
            global_seed = seed or kit_lib.gen_random_seed()
        else:
            global_seed = array_ops.zeros([1,], dtype=int64)
        re_seed = hvd.broadcast(global_seed, root_rank=0, name="random_seed")
        if (seed and seed != re_seed):
            logging.warning("The seed: {} is not consistent with that from cheif-node: {}, "
                            "and the seed from cheif-node will be used.".format(global_seed, re_seed))

        visible_devices = _get_visible_devices()
        status = kit_lib.plugin_init(local_rank, hvd.size(), unique_id, re_seed, 
                                     visible_devices, 
                                     global_batch_size=kwargs["global_batch_size"])
        return status

    def _one_device_init(**kwargs):
        """
        This function use to initialize only one GPU for SOK.
        """
        local_rank = 0
        unique_id = kit_lib.get_nccl_unique_id()
        global_seed = kwargs.get("seed", None) or kit_lib.gen_random_seed()
        visible_devices = _get_visible_devices()
        status = kit_lib.plugin_init(local_rank, 1, unique_id, global_seed, visible_devices,
                                     global_batch_size=kwargs["global_batch_size"])
        return status

    if tf_dist.has_strategy():
        strategy = tf_dist.get_strategy()

        @function
        def _init_wrapper(run_fn, init_fn, **kwargs):
            return run_fn(init_fn, kwargs=kwargs)

        if isinstance(strategy, MirroredStrategy):
            _init_fn = _single_worker_init
        elif isinstance(strategy, MultiWorkerMirroredStrategy):
            _init_fn = _multi_worker_init
        else:
            raise RuntimeError("This strategy type is not supported yet.")

        if not kit_lib.in_tensorflow2():
            _init_results = _init_wrapper(strategy.experimental_run_v2, _init_fn, **kwargs)
            if hasattr(_init_results, "values"): 
                _init_results =  _init_results.values
            return _init_results
        else:
            return _init_wrapper(strategy.run, _init_fn, **kwargs)
        
    elif "horovod.tensorflow" in sys.modules:
        # imported horovod
        import horovod.tensorflow as hvd

        if not kit_lib.in_tensorflow2():
            @function
            def _init_wrapper(**kwargs):
                return _horovod_init(**kwargs)
            return _init_wrapper(**kwargs)
        else:
            return _horovod_init(**kwargs)
    else:
        # horovod not imported
        return _one_device_init(**kwargs)
