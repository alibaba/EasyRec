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

import numpy as np
import tensorflow as tf
from tensorflow.python.distribute.values import PerReplica

import sys, os
sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), r"../../python/")))

import sok_unit_test_lib

class TestAllGatherDispatcher_single(SingleWorkerbase):
    def __init__(self):
        self.global_batch_size = 65536
        super(TestAllGatherDispatcher_single, self).__init__(global_batch_size=self.global_batch_size)

    def call(self):
        rows_num_per_sample = 26
        max_nnz = 3

        all_inputs = np.random.randint(low=1, high=100, size=[self.global_batch_size * rows_num_per_sample, max_nnz])

        all_mask = np.random.randint(low=0, high=2, size=[self.global_batch_size * rows_num_per_sample, max_nnz])

        all_inputs *= all_mask
        print("[INFO] original dense all inputs:\n", all_inputs)

        all_valid_indices = tf.where(all_inputs != 0)
        all_valid_values = tf.gather_nd(all_inputs, all_valid_indices)

        all_inputs_sparse_tensor = tf.sparse.SparseTensor(values=all_valid_values, indices=all_valid_indices, dense_shape=all_inputs.shape)
        print("[INFO] original inputs sparse tensor:\n", all_inputs_sparse_tensor)


        sparse_tensors = tf.sparse.split(sp_input=all_inputs_sparse_tensor, num_split=8, axis=0)
        sparse_tensors = PerReplica(sparse_tensors)
        print("[INFO] to each replica sparse tensors:\n", sparse_tensors)

        target_values = all_inputs_sparse_tensor.values
        # target_indices = tf.concat([tf.transpose(sparse_tensor.indices, perm=[1, 0])[0] 
        #                             for sparse_tensor in sparse_tensors.values],
        #                            axis=0)
        target_indices = tf.transpose(all_inputs_sparse_tensor.indices, perm=[1, 0])[0]
        target_num_elements = tf.concat([tf.shape(sparse_tensor.indices, out_type=tf.int64)[0] 
                                    for sparse_tensor in sparse_tensors.values],
                                    axis=0)
        target_total_valid_num = tf.size(target_values, out_type=tf.int64)
        print("[INFO] target_values: \n", target_values)
        print("[INFO] target_indcies: \n", target_indices)
        print("[INFO] target_num_elements: \n", target_num_elements)
        print("[INFO] target_total_valid_num: \n", target_total_valid_num)

        @tf.function
        def _step(sparse_tensor):
            if not isinstance(sparse_tensor, tf.sparse.SparseTensor):
                raise RuntimeError("sparse_tensor must be a tf.sparse.SparseTensor")

            values = sparse_tensor.values # [num_of_valids,]
            indices = sparse_tensor.indices 
            row_indices = tf.transpose(indices, perm=[1, 0])[0] # [num_of_valids]

            replica_ctx = tf.distribute.get_replica_context()

            values_out, indices_out, num_elements, total_valid_num = sok_unit_test_lib.all_gather_dispatcher(
                                                            replica_ctx.replica_id_in_sync_group,
                                                            replica_ctx.num_replicas_in_sync,
                                                            values,
                                                            row_indices,
                                                            global_batch_size=self.global_batch_size,
                                                            rows_num_per_sample=rows_num_per_sample,
                                                            max_nnz=max_nnz)
            return values_out, indices_out, num_elements, total_valid_num

        values_out, indices_out, num_elements, total_valid_num = self.strategy.run(_step, args=(sparse_tensors,))
        print("[INFO]: after all gather dispatcher, values = \n", values_out)
        print("[INFO]: after all gather dispatcher, indices = \n", indices_out)
        print("[INFO]: after all gather dispatcher, num_elements = \n", num_elements)
        print("[INFO]: after all gather dispatcher, total_valid_num = \n", total_valid_num)

        for i in range(len(values_out.values)):
            tf.debugging.assert_equal(target_values, values_out.values[i][:target_values.shape[0]], 
                                        message="values %d not meet target." %i)
            tf.debugging.assert_equal(target_indices, indices_out.values[i][:target_indices.shape[0]], 
                                        message="indcies %d not meet target." %i)
            tf.debugging.assert_equal(target_num_elements, num_elements.values[i][:target_num_elements.shape[0]], 
                                        message="num_elements %d not meet target." %i)
            tf.debugging.assert_equal(target_total_valid_num, total_valid_num.values[i],
                                        message="total_valid_num %d not meet target." %i)


if __name__ == "__main__":
    TestAllGatherDispatcher_single()()