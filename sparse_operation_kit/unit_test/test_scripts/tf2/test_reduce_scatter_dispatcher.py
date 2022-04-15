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

from single_worker_base import SingleWorkerbase
from multi_worker_base import MultiWorkerbase

import numpy as np
import tensorflow as tf
from tensorflow.python.distribute.values import PerReplica

import sys, os
sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), r"../../python/")))
import sok_unit_test_lib

class TestReduceScatterDispatcher_single(SingleWorkerbase):
    def __init__(self):
        self.global_batch_size = 65536
        super(TestReduceScatterDispatcher_single, self).__init__(global_batch_size=self.global_batch_size)

    def call(self):
        slot_num = 26
        max_nnz = 3
        embedding_vec_size = 16

        input_in_each_gpu = [np.random.normal(size=(self.global_batch_size * slot_num, embedding_vec_size)).astype(np.float32)
                             for _ in range(8)]
        input_in_each_gpu_ = PerReplica(input_in_each_gpu)
        print("[INFO] original inputs: \n", input_in_each_gpu_)

        target = tf.split(tf.reshape(tf.math.reduce_sum(input_in_each_gpu, axis=0), 
                                     shape=[self.global_batch_size, slot_num, embedding_vec_size]), 
                          num_or_size_splits=8)
        print("[INFO] target output: \n", target)

        @tf.function
        def _step(replica_input):
            replica_ctx = tf.distribute.get_replica_context()

            replica_output = sok_unit_test_lib.reduce_scatter_dispatcher(replica_ctx.replica_id_in_sync_group,
                                                                            replica_input,
                                                                            global_batch_size=self.global_batch_size,
                                                                            slot_num=slot_num,
                                                                            max_nnz=max_nnz)
            return replica_output

        outputs = self.strategy.run(_step, args=(input_in_each_gpu_,))
        print("[INFO] replica output:\n", outputs)

        for i in range(len(input_in_each_gpu)):
            tf.debugging.assert_near(target[i], outputs.values[i],
                                     message="output %d not meet target.",
                                     atol=1e-5,
                                     rtol=1e-5)


if __name__ == "__main__":
    TestReduceScatterDispatcher_single()()