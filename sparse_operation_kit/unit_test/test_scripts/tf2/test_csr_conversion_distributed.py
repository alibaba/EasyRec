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

class CreateDataset(object):
    def __init__(self, 
                 dataset_names,
                 feature_desc,
                 batch_size,
                 n_epochs,
                 slot_num,
                 max_nnz,
                 convert_to_csr=False,
                 gpu_count=1,
                 embedding_type='localized',
                 get_row_indices=False):
        self.dataset_names = dataset_names
        self.feature_desc = feature_desc
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.slot_num = slot_num
        self.max_nnz = max_nnz
        self.convert_to_csr = convert_to_csr
        self.gpu_count = gpu_count
        self.embedding_type = embedding_type
        self.get_row_indices = get_row_indices

        if (self.convert_to_csr and self.get_row_indices):
            raise RuntimeError("convert_to_csr and get_row_indices could not be True at the same time.")

        self.num_threads = 32


    def __call__(self):
        dataset = tf.data.TFRecordDataset(filenames=self.dataset_names, compression_type=None,
                                            buffer_size=100 * 1024 * 1024, num_parallel_reads=self.num_threads)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.repeat(self.n_epochs)

        dataset = dataset.map(lambda serialized: self._parse_fn(serialized),
                              num_parallel_calls=1,
                              deterministic=False)

        dataset = dataset.prefetch(buffer_size=16)
        
        return dataset

    
    @tf.function
    def _parse_fn(self, serialized):
        with tf.name_scope("datareader_map"):
            features = tf.io.parse_example(serialized, self.feature_desc)

            label = features['label']
            dense = tf.TensorArray(dtype=tf.int64, size=utils.NUM_INTEGER_COLUMNS, dynamic_size=False,
                                   element_shape=(self.batch_size,))
            cate = tf.TensorArray(dtype=tf.int64, size=utils.NUM_CATEGORICAL_COLUMNS, dynamic_size=False,
                                    element_shape=(self.batch_size, 1))

            for idx in range(utils.NUM_INTEGER_COLUMNS):
                dense = dense.write(idx, features['I' + str(idx + 1)])

            for idx in range(utils.NUM_CATEGORICAL_COLUMNS):
                cate = cate.write(idx, features['C' + str(idx + 1)])

            dense = tf.transpose(dense.stack(), perm=[1, 0])
            cate = tf.transpose(cate.stack(), perm=[1, 0, 2])

            if self.convert_to_csr:
                row_offsets, value_tensors, nnz_array = self._distribute_keys(all_keys=cate)
                
                place_holder = tf.sparse.SparseTensor([[0,0]], tf.constant([0], dtype=tf.int64), 
                                                      [self.batch_size * utils.NUM_CATEGORICAL_COLUMNS,1])

                return label, dense, row_offsets, value_tensors, nnz_array, place_holder
            else:
                reshape_keys = tf.reshape(cate, [-1, self.max_nnz])
                indices = tf.where(reshape_keys != -1)
                values = tf.gather_nd(reshape_keys, indices)
                sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=reshape_keys.shape)
                place_holder = tf.constant(1, dtype=tf.int64)

                if self.get_row_indices:
                    row_indices = tf.transpose(indices, perm=[1, 0])[0]
                    return label, dense, row_indices, values, place_holder, sparse_tensor
                else:
                    return label, dense, place_holder, place_holder, place_holder, sparse_tensor

    @tf.function
    def _distribute_keys(self, all_keys):
        return tf.cond(tf.equal("distributed", self.embedding_type),
                        lambda: self._distribute_keys_for_distributed(all_keys),
                        lambda: self._distribute_keys_for_localized(all_keys)) 

    @tf.function
    def _localized_recompute_row_indices(self, row_indices, slot_mod, dev_id):
        batch_idx = row_indices // self.slot_num
        slot_idx = row_indices % self.slot_num
        dev_slot_idx = slot_idx // self.gpu_count
        dev_slot_num = tf.cast(self.slot_num // self.gpu_count + (1 if dev_id < slot_mod else 0), dtype=batch_idx.dtype)
        dev_row_indices = batch_idx * dev_slot_num + dev_slot_idx
        return dev_row_indices

    @tf.function
    def _distribute_keys_for_localized(self, all_keys):
        slot_mod = tf.cast(self.slot_num % self.gpu_count, dtype=tf.int32)
        reshape_keys = tf.reshape(all_keys, [self.batch_size * self.slot_num, self.max_nnz])
        
        valid_indices = tf.where(reshape_keys != 0)
        valid_keys = tf.gather_nd(reshape_keys, valid_indices)
        coo_indices = tf.transpose(valid_indices, perm=[1, 0])

        slot_dev_idx = tf.cast(coo_indices[0] % self.slot_num, dtype=tf.int32)

        roof_slot_num_gpu_count = self.slot_num // self.gpu_count
        roof_slot_num_gpu_count += (1 if self.slot_num % self.gpu_count != 0 else 0)

        row_offsets = tf.TensorArray(dtype=tf.int64, size=self.gpu_count, dynamic_size=False,
                                     clear_after_read=False, 
                                     element_shape=[self.batch_size * (roof_slot_num_gpu_count) + 1])
        value_tensors = tf.TensorArray(dtype=tf.int64, size=self.gpu_count, dynamic_size=False,
                                    clear_after_read=False,
                                    element_shape=[self.batch_size * (roof_slot_num_gpu_count) * self.max_nnz])
        nnz_array = tf.TensorArray(dtype=tf.int64, size=self.gpu_count, dynamic_size=False, clear_after_read=False)

        for dev_id in tf.range(self.gpu_count, dtype=tf.int32):
            flag_indices = tf.where(slot_dev_idx % self.gpu_count == dev_id)
            row_indexes = tf.gather_nd(coo_indices[0], flag_indices)

            # recompute dev row_idexes in each GPU
            row_indexes = self._localized_recompute_row_indices(row_indexes, slot_mod, dev_id)

            col_indexes = tf.gather_nd(coo_indices[1], flag_indices)
            dev_keys = tf.gather_nd(valid_keys, flag_indices)

            sparse_indices = tf.transpose(tf.stack([row_indexes, col_indexes]), perm=[1, 0])
            csr_sparse_matrix = tf.raw_ops.SparseTensorToCSRSparseMatrix(indices=sparse_indices, 
                                                                         values=tf.cast(dev_keys, dtype=tf.float64),
                                dense_shape=tf.cond(dev_id < slot_mod,
                                    lambda: (self.batch_size * ((self.slot_num // self.gpu_count) + 1), self.max_nnz),
                                    lambda: (self.batch_size * (self.slot_num // self.gpu_count), self.max_nnz)))

            row_ptrs, _, _ = tf.raw_ops.CSRSparseMatrixComponents(csr_sparse_matrix=csr_sparse_matrix,
                                                                  index=0,
                                                                  type=tf.float64)

            row_ptrs = tf.cast(row_ptrs, dtype=tf.int64)
            nnz_array = nnz_array.write(dev_id, row_ptrs[-1])
            row_ptrs = tf.pad(row_ptrs, paddings=[[0, self.batch_size * (roof_slot_num_gpu_count) + 1 - tf.shape(row_ptrs)[0]]])
            values = tf.pad(dev_keys, paddings=[[0, self.batch_size * (roof_slot_num_gpu_count) * self.max_nnz - tf.shape(dev_keys)[0]]])
            row_offsets = row_offsets.write(dev_id, row_ptrs)
            value_tensors = value_tensors.write(dev_id, values)

        return row_offsets.stack(), value_tensors.stack(), nnz_array.stack()
        

    @tf.function
    def _distribute_keys_for_distributed(self, all_keys):
        reshape_keys = tf.reshape(all_keys, [self.batch_size * self.slot_num, self.max_nnz])

        valid_indices = tf.where(reshape_keys != 0)
        valid_values = tf.gather_nd(reshape_keys, valid_indices)
        coo_indices = tf.transpose(valid_indices, perm=[1, 0])

        row_offsets = tf.TensorArray(dtype=tf.int64, size=self.gpu_count, dynamic_size=False,
                                    clear_after_read=True)
        value_tensors = tf.TensorArray(dtype=tf.int64, size=self.gpu_count, dynamic_size=False,
                                    element_shape=[self.batch_size * self.slot_num * self.max_nnz],
                                    clear_after_read=True)
        nnz_array = tf.TensorArray(dtype=tf.int64, size=self.gpu_count, dynamic_size=False, 
                                   clear_after_read=True)

        for dev_id in tf.range(self.gpu_count, dtype=tf.int32):
            binary_indices = tf.where(tf.cast(valid_values % self.gpu_count, dtype=tf.int32) == dev_id)
            row_indexes = tf.gather_nd(coo_indices[0], binary_indices)
            col_indexes = tf.gather_nd(coo_indices[1], binary_indices)
            dev_values = tf.gather_nd(valid_values, binary_indices)

            sparse_indices = tf.transpose(tf.stack([row_indexes, col_indexes]), perm=[1, 0])
            csr_sparse_matrix = tf.raw_ops.SparseTensorToCSRSparseMatrix(indices=sparse_indices,
                                                        values=tf.cast(dev_values, dtype=tf.float64),
                                            dense_shape=(self.batch_size * self.slot_num, self.max_nnz))

            row_ptrs, _, _ = tf.raw_ops.CSRSparseMatrixComponents(csr_sparse_matrix=csr_sparse_matrix, index=0, type=tf.float64)
            dev_values = tf.pad(dev_values, paddings=[[0, self.batch_size * self.slot_num * self.max_nnz - tf.shape(dev_values)[0]]])

            row_ptrs = tf.cast(row_ptrs, dtype=tf.int64)
            row_offsets = row_offsets.write(dev_id, row_ptrs)
            value_tensors = value_tensors.write(dev_id, dev_values)
            nnz_array = nnz_array.write(dev_id, row_ptrs[-1])

        return row_offsets.stack(), value_tensors.stack(), nnz_array.stack()



class TestCsrConversionDistributed_single(SingleWorkerbase):
    def __init__(self):
        self.global_batch_size = 65536
        super(TestCsrConversionDistributed_single, self).__init__(global_batch_size=self.global_batch_size)

    def call(self):
        slot_num = 26
        max_nnz = 10

        all_inputs = np.random.randint(low=1, high=1000, size=[self.global_batch_size * slot_num, max_nnz])

        all_mask = np.random.randint(low=0, high=2, size=[self.global_batch_size * slot_num, max_nnz])

        all_inputs *= all_mask
        print("[INFO] original all_inputs:\n", all_inputs)

        all_valid_indices = tf.where(all_inputs != 0)
        all_valid_values = tf.gather_nd(all_inputs, all_valid_indices)

        all_inputs_sparse_tensor = tf.sparse.SparseTensor(values=all_valid_values, indices=all_valid_indices, dense_shape=all_inputs.shape)
        print("[INFO] original inputs sparse tensor:\n", all_inputs_sparse_tensor)

        data_obj = CreateDataset(dataset_names=None, feature_desc=None, batch_size=self.global_batch_size,
                                 n_epochs=None, slot_num=slot_num, max_nnz=max_nnz, convert_to_csr=True,
                                 gpu_count=8, embedding_type="distributed", get_row_indices=False)

        row_offsets, values_tensors, nnz_array = data_obj._distribute_keys_for_distributed(all_inputs)
        print("[INFO]: csr target values:\n", values_tensors)
        print("[INFO]: csr target row_offsets:\n", row_offsets)
        print("[INFO]: csr target nnz_array:\n", nnz_array)

        @tf.function
        def _step(sparse_tensor):
            if not isinstance(sparse_tensor, tf.sparse.SparseTensor):
                raise RuntimeError("sparse_tensor must be a tf.sparse.SparseTensor")

            values = sparse_tensor.values
            indices = sparse_tensor.indices
            row_indices = tf.transpose(indices, perm=[1, 0])[0]

            replica_ctx = tf.distribute.get_replica_context()

            replica_values, replica_csr_row_offsets, replica_nnz = sok_unit_test_lib.csr_conversion_distributed(
                                replica_ctx.replica_id_in_sync_group,
                                values,
                                row_indices,
                                tf.size(values, out_type=tf.int64),
                                global_batch_size=self.global_batch_size,
                                slot_num=slot_num,
                                max_nnz=max_nnz)
            return replica_values, replica_csr_row_offsets, replica_nnz

        replica_values, replica_csr_row_offsets, replica_nnz = self.strategy.run(_step, args=(all_inputs_sparse_tensor,))
        print("[INFO]: after conversion, repllica values = \n", replica_values)
        print("[INFO]: after conversion, replica csr row_offset = \n", replica_csr_row_offsets)
        print("[INFO]: after conversion, replica nnz = \n", replica_nnz)

        for i in range(values_tensors.shape[0]):
            tf.debugging.assert_equal(values_tensors[i], replica_values.values[i],
                                      message="values %d not meet target" %i)
            tf.debugging.assert_equal(row_offsets[i], replica_csr_row_offsets.values[i],
                                      message="csr row_offset %d not meet target" %i)
            tf.debugging.assert_equal(nnz_array[i], replica_nnz.values[i], 
                                      message="nnz %d not meet target" %i)


if __name__ == "__main__":
    TestCsrConversionDistributed_single()()

