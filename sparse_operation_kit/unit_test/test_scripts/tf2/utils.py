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

import sys, os
sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), r"../../../")))
import sparse_operation_kit as sok
import tensorflow as tf

import pickle
import numpy as np
from multiprocessing import Process

local_ips = ("localhost", "127.0.0.1", "0.0.0.0")

def get_local_ip(hostname=None):
    import socket
    _hostname = socket.gethostname()
    return socket.gethostbyname(hostname or socket.gethostname())

def is_local_ip(ip_address):
    return True if ip_address in local_ips else False

def all_ips_in_local(ips):
    for ip in ips:
        if not is_local_ip(ip):
            return False
    return True

def get_local_gpu_count():
    import os
    text = os.popen("nvidia-smi --list-gpus").read()
    text = text.strip().split("\n")
    return len(text)
    
def get_cuda_version():
    import os, re
    text = os.popen("nvcc --version").read()
    version = text.strip().split("\n")[-1]
    version = re.search("cuda_\d+.\d+.", version).group(0)
    version = re.search("\d+.\d+", version).group(0)
    return version

class TestProcess(object):
    def __init__(self,
                 func,
                 task_id,
                 arguments):
        self.func = func
        self.task_id = task_id
        self.arguments = arguments
        self.arguments.task_id = self.task_id

        self.process = Process(target=self.func, args=(self.arguments,))

    def start(self):
        self.process.start()

    def join(self):
        if self.process.is_alive():
            self.process.join()


def save_to_file(filename, *args):
    with open(filename, 'wb') as file:
        num_of_items = len(args)
        if (num_of_items == 0):
            raise ValueError("Nothing needed to be saved.")
        pickle.dump(num_of_items, file, pickle.HIGHEST_PROTOCOL)
        for item in args:
            pickle.dump(item, file, pickle.HIGHEST_PROTOCOL)
    print("[INFO]: dumpped items to file %s" %filename)

def restore_from_file(filename):
    results = list()
    with open(filename, "rb") as file:
        num_of_items = pickle.load(file)
        for _ in range(num_of_items):
            item = pickle.load(file)
            results.append(item)
    print("[INFO] loadded from file %s" %filename)
    return tuple(results)
    
def get_embedding_optimizer(optimizer_type):
    if not isinstance(optimizer_type, str):
        raise ValueError("optimizer_type must be str type, but got ", type(optimizer_type))
    if optimizer_type == "plugin_adam":
        return sok.optimizers.Adam
    elif optimizer_type == 'adam':
        return tf.keras.optimizers.Adam
    elif optimizer_type == 'sgd':
        return tf.keras.optimizers.SGD
    else:
        raise ValueError("Not supported optimizer_type: %s" %optimizer_type)

def get_dense_optimizer(optimizer_type):
    if not isinstance(optimizer_type, str):
        raise ValueError("optimizer_type must be str type, but got ", type(optimizer_type))
    if optimizer_type == "plugin_adam":
        return tf.keras.optimizers.Adam
    elif optimizer_type == 'adam':
        return tf.keras.optimizers.Adam
    elif optimizer_type == 'sgd':
        return tf.keras.optimizers.SGD
    else:
        raise ValueError("Not supported optimizer_type: %s" %optimizer_type)

def get_ones_tensor(max_vocab_size_per_gpu,
                    embedding_vec_size,
                    num,
                    task_id=None):
    tensor = np.ones(shape=[max_vocab_size_per_gpu, embedding_vec_size], dtype=np.float32)
    all_tensors = [tensor for _ in range(num)]

    return all_tensors

def get_random_value(shape, dtype=None):
    tensor = np.random.normal(size=shape)
    tensor = tensor.astype(np.float32)
    return tensor


def generate_random_samples(num_of_samples,
                            vocabulary_size,
                            slot_num, 
                            max_nnz,
                            dtype=np.int64,
                            use_sparse_mask=True):
    """
    This function is used to generate random samples used for training.
    #args:
        num_of_samples: integer, how many samples should be generated.
        vocabulary_size: integer,
        slot_num: integer,
        max_nnz: integer
        use_sparse_mask: boolean, whether to use sparse mask to generate sparse datas
    #returns: 
        all_keys: dense tensor, whose shape is [num_of_samples, slot_num, max_nnz]
        all_labels: dense tensor, whose shape is [num_of_samples, 1]
    """
    print("[INFO]: begin to generate random samples")
    
    from tensorflow.python.distribute.values import PerReplica
    cuda_version = get_cuda_version()
    cuda_version = "".join(cuda_version.split("."))
    try:
        import cupy as cp
    except:
        import os
        os.system("pip install cupy-cuda"+cuda_version)
        import cupy as cp

    if (vocabulary_size // slot_num <= 2 * max_nnz):
        raise ValueError("Too small vocabulary_size. vocabulary_size: %d // slot_num: %d = %d <= 2 * max_nnz: %d"
                        %(vocabulary_size, slot_num, vocabulary_size // slot_num, 2 * max_nnz))

    if use_sparse_mask:
        mask = np.random.choice([-1, 1], size=(num_of_samples, slot_num, max_nnz))
        filter_ = np.ones(shape=(num_of_samples, slot_num, max_nnz))
        sum_ = np.sum(mask * filter_, axis=-1, keepdims=True)
        index = np.where(sum_ == -max_nnz)
        index = tuple(map(lambda array: array[1:] if array.ndim and array.size else array, index))
        mask[index] = 1

    with cp.cuda.Device(0):
        all_keys = cp.zeros(shape=(num_of_samples, slot_num, max_nnz), dtype=cp.int64)
        
        random_kernel = cp.RawKernel(r'''
            __device__ size_t randInt(size_t gid, const size_t range) {
                return (((gid * clock() * 214013L + 2531011L) >> 16) & 0x7fff) % range;
            }

            extern "C" __global__ 
            void my_kernel(long long *nums, const size_t count,
                        const size_t slot_num, const size_t max_nnz,
                        const size_t vocab_per_slot) {
                size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
                for (size_t i = gid; i < count; i += blockDim.x * gridDim.x) {
                    size_t tid_in_sample = i % (slot_num * max_nnz);
                    size_t slot_id = tid_in_sample / max_nnz;
                    size_t col_id = tid_in_sample % max_nnz;
                    nums[i] = vocab_per_slot * slot_id + randInt(gid, vocab_per_slot);
                }
            }
        ''', 'my_kernel')

        random_kernel((num_of_samples,), (1024,), 
                    (all_keys, num_of_samples * slot_num * max_nnz,
                    slot_num, max_nnz, vocabulary_size // slot_num))
        all_keys = all_keys.get()

    if use_sparse_mask:
        all_keys[mask == -1] = -1

    all_keys = np.sort(all_keys, axis=-1)[:,:,::-1]    
    
    all_labels = np.random.randint(low=0, high=2, size=(num_of_samples, 1))

    print("[INFO]: generated random samples")
    return all_keys, all_labels

def tf_dataset(keys, labels,
               batchsize,
               to_sparse_tensor=False,
               repeat=None,
               args=None):

    num_of_samples, slot_num, max_nnz = keys.shape
    def _convert_to_sparse(keys, labels):
        if tf.rank(keys) != 2:
            keys = tf.reshape(keys, shape=[-1, max_nnz])
        indices = tf.where(keys != -1)
        values = tf.gather_nd(keys, indices)
        if args is not None and hasattr(args, "key_dtype"):
            if args.key_dtype == "int64":
                values = tf.cast(values, dtype=tf.int64)
            elif args.key_dtype == "uint32":
                values = tf.cast(values, dtype=tf.uint32)
            else:
                raise ValueError("Not supported key_dtype.")
        return tf.sparse.SparseTensor(indices=indices, 
                                      values=values, 
                                      dense_shape=[batchsize * slot_num, max_nnz]), labels
    def _cast_values(keys, labels):
        if args is not None and hasattr(args, "key_dtype"):
            if args.key_dtype == "int64":
                keys = tf.cast(keys, dtype=tf.int64)
            elif args.key_dtype == "uint32":
                keys = tf.cast(keys, dtype=tf.uint32)
            else:
                raise ValueError("Not supported key_dtype.")
        return keys, labels

    dataset = tf.data.Dataset.from_tensor_slices((keys, labels))
    dataset = dataset.repeat(repeat)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    if to_sparse_tensor:
        dataset = dataset.map(lambda keys, labels: 
                                _convert_to_sparse(keys, labels),
                            num_parallel_calls=1)
    else:
        dataset = dataset.map(lambda keys, labels:
                                _cast_values(keys, labels),
                            num_parallel_calls=1)
    return dataset

def try_make_dirs(directory, chief=True):
    import os
    if not os.path.exists(directory) and chief:
        os.makedirs(directory)

def sort_embedding_variables_by_key(keys, embedding_values, embedding_vec_size, use_hashtable=True, gpu_num=None):
    """
    This function is used to sort the embedding values by its relavent keys.
    For example, keys: [5, 3, 6, 1], embedding values: [[0, 0, 0, 0],
                                                        [1, 1, 1, 1],
                                                        [2, 2, 2, 2],
                                                        [3, 3, 3, 3]]
    After sorted, keys: [1, 3, 5, 6], embedding values: [[3, 3, 3, 3],
                                                         [1, 1, 1, 1],
                                                         [0, 0, 0, 0],
                                                         [2, 2, 2, 2]]
    """
    cuda_version = get_cuda_version()
    cuda_version = "".join(cuda_version.split("."))
    try:
        import cupy as cp
    except:
        import os
        os.system("pip install cupy-cuda"+cuda_version)
        import cupy as cp

    if not isinstance(keys, np.ndarray):
        keys = np.array(keys, dtype=np.int64)
    if not isinstance(embedding_values, np.ndarray):
        embedding_values = np.array(embedding_values, dtype=np.float32)

    # currently, embedding will set a fast hashtable when user specified use_hashtable=False
    # so that the following code snippet is not needed.
    """
    if not use_hashtable:
        vocabulary_size = np.size(keys) // gpu_num
        embedding_values = np.reshape(embedding_values, newshape=(-1, embedding_vec_size))
        embedding_values_list = np.split(embedding_values, gpu_num, axis=0)
        for gpu_id, emb_values in enumerate(embedding_values_list):
            invalid_keys = np.array([key for key in range(vocabulary_size) if key % gpu_num != gpu_id], dtype=np.int64)
            emb_values[invalid_keys] = 0
        valid_embedding_values = np.sum(embedding_values_list, axis=0)
        return keys[:vocabulary_size], valid_embedding_values
    else:
        del gpu_num
    """

    sorted_indexes = np.argsort(keys)
    sorted_keys = keys[sorted_indexes]
    
    with cp.cuda.Device(0):
        d_sorted_values = cp.zeros(shape=embedding_values.shape, dtype=cp.float32)
        d_sorted_indexes = cp.asarray(sorted_indexes)
        d_embedding_values = cp.asarray(embedding_values)

        sort_values_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void my_kernel(const size_t *sorted_indexes, 
                           const float *values,
                           float *sorted_values, 
                           const size_t values_step, 
                           const size_t count) {
                const size_t col_id = threadIdx.x;
                for (size_t row_id = blockIdx.x; row_id < count; row_id += blockDim.x) {
                    sorted_values[row_id * values_step + col_id] =
                            values[sorted_indexes[row_id] * values_step + col_id];
                }
            } 
        ''', 'my_kernel')

        sort_values_kernel((keys.size,), (embedding_vec_size,),
                            (d_sorted_indexes, d_embedding_values, d_sorted_values, 
                            embedding_vec_size, keys.size))
        sorted_values = d_sorted_values.get()

    return sorted_keys, sorted_values

def read_binary_file(filename,
                     element_type,
                     chunk_num_elements=65536):
    import struct, os

    element_type_map = {"float": ["f", 4],
                        "int32": ["i", 4],
                        "long long": ["q", 8],
                        "unsigned long long": ["Q", 8],
                        "size_t": ["N", 8],
                        "unsigned int": ["I", 4]}

    elem_size_in_bytes = element_type_map[element_type][1]

    file_size_in_bytes = os.path.getsize(filename)
    if (file_size_in_bytes % elem_size_in_bytes != 0):
        raise ValueError("Invalid element size for file: %s." %filename)

    chunk_size_in_bytes = chunk_num_elements * elem_size_in_bytes
    if (file_size_in_bytes <= chunk_size_in_bytes):
        chunk_size_in_bytes = file_size_in_bytes
        chunk_count = 1
    else:
        chunk_count = file_size_in_bytes // chunk_size_in_bytes

    results = list()
    with open(filename, "rb") as file:
        for _ in range(chunk_count):
            buffer = file.read(chunk_size_in_bytes)
            if (0 == len(buffer)):
                raise RuntimeError("Error in reading file.")
            elements = struct.unpack(str(chunk_size_in_bytes // elem_size_in_bytes) + 
                                     element_type_map[element_type][0],
                                     buffer)        
            results += elements
        
        if (file_size_in_bytes - chunk_count * chunk_size_in_bytes > 0):
            buffer_size_in_bytes = file_size_in_bytes - chunk_count * chunk_size_in_bytes
            buffer = file.read(buffer_size_in_bytes)
            elements = struct.unpack(str(buffer_size_in_bytes // elem_size_in_bytes) + 
                                     element_type_map[element_type][0],
                                     buffer)
            results += elements
    return results

def get_valid_tf_values(keys, values):
    if not isinstance(keys, np.ndarray):
        keys = np.array(keys, dtype=np.int64)
    if not isinstance(values, np.ndarray):
        values = np.array(values, dtype=np.float32)

    keys = tf.reshape(keys, [-1])
    return tf.gather(values, keys).numpy()

if __name__ == "__main__":
    all_keys, all_labels = generate_random_samples(num_of_samples=65536 * 100,
                                                   vocabulary_size=8 * 1024 * 1,
                                                   slot_num=10,
                                                   max_nnz=4,
                                                   use_sparse_mask=False)

    # print("all_keys:\n", all_keys)
    # print("all_labels:\n", all_labels)

    dataset = tf_dataset(keys=all_keys, labels=all_labels,
                         batchsize=65536, 
                         to_sparse_tensor=False,
                         repeat=1)
    for i, (input_tensors, labels) in enumerate(dataset):
        print("-"*30, "Iteration ", str(i), "-"*30)
        print(input_tensors)
        print(labels)


    # a = [1, 2, 3]
    # b = [4, 5]
    # save_to_file("./test.file", a, b)
    # a = restore_from_file("./test.file")
    # print(a)

    # local_ip = get_local_ip()
    # print("local_ip: %s" %local_ip)

    # keys = np.array([5, 3, 6, 1], dtype=np.int64)
    # values = np.array([[0, 0, 0, 0],
    #                    [1, 1, 1, 1],
    #                    [2, 2, 2, 2],
    #                    [3, 3, 3, 3]], dtype=np.float32)

    # sorted_keys, sorted_values = sort_embedding_variables_by_key(keys, values, embedding_vec_size=4)
    # print(sorted_keys)
    # print(sorted_values)

    # filename = r"./embedding_variables/test_values.file"
    # keys = read_binary_file(filename, element_type="float")
    # print(len(keys))

    # keys = [5, 3, 6, 1]
    # values = [[0, 0],
    #           [1, 1],
    #           [2, 2],
    #           [3, 3],
    #           [4, 4], 
    #           [5, 5],
    #           [6, 6]]
    # print(get_valid_tf_values(keys, values))


