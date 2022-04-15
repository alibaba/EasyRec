import os
import queue
import concurrent

import numpy as np
import tensorflow as tf


class BinaryDataset:

    def __init__(
        self,
        label_bin,
        dense_bin,
        category_bin,
        batch_size=1,
        drop_last=True,
        global_rank=0,
        global_size=1,
        prefetch=1,
        label_raw_type=np.int32,
        dense_raw_type=np.int32,
        category_raw_type=np.int32,
        log=True,
    ):
        """
        * batch_size : The batch size of local rank, which means the total batch size of all ranks should be (batch_size * global_size).
        * prefetch   : If prefetch > 1, it can only be read sequentially, otherwise it will return incorrect samples.
        """
        self._check_file(label_bin, dense_bin, category_bin)

        self._batch_size = batch_size
        self._drop_last = drop_last
        self._global_rank = global_rank
        self._global_size = global_size

        # actual number of samples in the binary file
        self._samples_in_all_ranks = os.path.getsize(label_bin) // 4

        # self._num_entries represents there are how many batches
        self._num_entries = self._samples_in_all_ranks // (batch_size * global_size)

        # number of samples in current rank
        self._num_samples = self._num_entries * batch_size

        if not self._drop_last:
            if (self._samples_in_all_ranks % (batch_size * global_size)) < global_size:
                print("The number of samples in last batch is less than global_size, so the drop_last=False will be ignored.")
                self._drop_last = True
            else:
                # assign the samples in the last batch to different local ranks
                samples_in_last_batch = [(self._samples_in_all_ranks % (batch_size * global_size)) // global_size] * global_size
                for i in range(global_size):
                    if i < (self._samples_in_all_ranks % (batch_size * global_size)) % global_size:
                        samples_in_last_batch[i] += 1
                assert(sum(samples_in_last_batch) == (self._samples_in_all_ranks % (batch_size * global_size)))
                self._samples_in_last_batch = samples_in_last_batch[global_rank]

                # the offset of last batch
                self._last_batch_offset = []
                offset = 0
                for i in range(global_size):
                    self._last_batch_offset.append(offset)
                    offset += samples_in_last_batch[i]

                # correct the values when drop_last=Fasle
                self._num_entries += 1
                self._num_samples = (self._num_entries - 1) * batch_size + self._samples_in_last_batch

        self._prefetch = min(prefetch, self._num_entries)
        self._prefetch_queue = queue.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._label_file = os.open(label_bin, os.O_RDONLY)
        self._dense_file = os.open(dense_bin, os.O_RDONLY)
        self._category_file = os.open(category_bin, os.O_RDONLY)

        self._label_raw_type = label_raw_type
        self._dense_raw_type = dense_raw_type
        self._category_raw_type = category_raw_type

        self._log = log

    def _check_file(self, label_bin, dense_bin, category_bin):
        # num_samples represents the actual number of samples in the dataset
        num_samples = os.path.getsize(label_bin) // 4
        if num_samples <= 0:
            raise RuntimeError("There must be at least one sample in %s"%label_bin)

        # check file size
        for file, bytes_per_sample in [[label_bin, 4], [dense_bin, 52], [category_bin, 104]]:
            file_size = os.path.getsize(file)
            if file_size % bytes_per_sample != 0:
                raise RuntimeError("The file size of %s should be an integer multiple of %d."%(file, bytes_per_sample))
            if (file_size // bytes_per_sample) != num_samples:
                raise RuntimeError("The number of samples in %s is not equeal to %s"%(dense_bin, label_bin))

    def __del__(self):
        for file in [self._label_file, self._dense_file, self._category_file]:
            if file is not None:
                os.close(file)

    def __len__(self):
        return self._num_entries

    def __getitem__(self, idx):
        if idx >= self._num_entries:
            raise IndexError()

        if self._prefetch <= 1:
            return self._get(idx)

        if idx == 0:
            for i in range(self._prefetch):
                self._prefetch_queue.put(self._executor.submit(self._get, (i)))

        if idx < (self._num_entries - self._prefetch):
            self._prefetch_queue.put(self._executor.submit(self._get, (idx + self._prefetch)))

        return self._prefetch_queue.get().result()

    def _get(self, idx):
        # calculate the offset & number of the samples to be read
        if not self._drop_last and idx == self._num_entries - 1:
            sample_offset = idx * (self._batch_size * self._global_size) + self._last_batch_offset[self._global_rank]
            batch = self._samples_in_last_batch
        else:
            sample_offset = idx * (self._batch_size * self._global_size) + (self._batch_size * self._global_rank)
            batch = self._batch_size

        # read the data from binary file
        label_raw_data = os.pread(self._label_file, 4 * batch, 4 * sample_offset)
        label = np.frombuffer(label_raw_data, dtype=self._label_raw_type).reshape([batch, 1])

        dense_raw_data = os.pread(self._dense_file, 52 * batch, 52 * sample_offset)
        dense = np.frombuffer(dense_raw_data, dtype=self._dense_raw_type).reshape([batch, 13])

        category_raw_data = os.pread(self._category_file, 104 * batch, 104 * sample_offset)
        category = np.frombuffer(category_raw_data, dtype=self._category_raw_type).reshape([batch, 26])

        # convert numpy data to tensorflow data
        if self._label_raw_type == self._dense_raw_type and self._label_raw_type == self._category_raw_type:
            data = np.concatenate([label, dense, category], axis=1)
            data = tf.convert_to_tensor(data)
            label = tf.cast(data[:, 0:1], dtype=tf.float32)
            dense = tf.cast(data[:, 1:14], dtype=tf.float32)
            category = tf.cast(data[:, 14:40], dtype=tf.int64)
        else:
            label = tf.convert_to_tensor(label, dtype=tf.float32)
            dense = tf.convert_to_tensor(dense, dtype=tf.float32)
            category = tf.convert_to_tensor(category, dtype=tf.int64)

        # preprocess
        if self._log:
            dense = tf.math.log(dense + 3.0)

        return (dense, category), label


class SplitedBinaryDataset:

    def __init__(
        self,
        label_bin,
        dense_bin,
        category_bin,
        vocab_sizes,
        batch_size=1,
        drop_last=True,
        global_rank=0,
        global_size=1,
        prefetch=1,
        label_raw_type=None,
        dense_raw_type=None,
        category_raw_type=None,
        log=None,
    ):
        """
        * category_bin      : The list of category binary files.
        * batch_size        : The batch size of local rank, which means the total batch size of all ranks should be (batch_size * global_size).
        * prefetch          : If prefetch > 1, it can only be read sequentially, otherwise it will return incorrect samples.
        * label_raw_type    : Deprecated.
        * dense_raw_type    : Deprecated.
        * category_raw_type : Deprecated.
        * log               : Deprecated.
        """
        self._check_file(label_bin, dense_bin, category_bin, vocab_sizes)

        self._batch_size = batch_size
        self._drop_last = drop_last
        self._global_rank = global_rank
        self._global_size = global_size

        # actual number of samples in the binary file
        self._samples_in_all_ranks = os.path.getsize(label_bin)

        # self._num_entries represents there are how many batches
        self._num_entries = self._samples_in_all_ranks // (batch_size * global_size)

        # number of samples in current rank
        self._num_samples = self._num_entries * batch_size

        if not self._drop_last:
            if (self._samples_in_all_ranks % (batch_size * global_size)) < global_size:
                print("The number of samples in last batch is less than global_size, so the drop_last=False will be ignored.")
                self._drop_last = True
            else:
                # assign the samples in the last batch to different local ranks
                samples_in_last_batch = [(self._samples_in_all_ranks % (batch_size * global_size)) // global_size] * global_size
                for i in range(global_size):
                    if i < (self._samples_in_all_ranks % (batch_size * global_size)) % global_size:
                        samples_in_last_batch[i] += 1
                assert(sum(samples_in_last_batch) == (self._samples_in_all_ranks % (batch_size * global_size)))
                self._samples_in_last_batch = samples_in_last_batch[global_rank]

                # the offset of last batch
                self._last_batch_offset = []
                offset = 0
                for i in range(global_size):
                    self._last_batch_offset.append(offset)
                    offset += samples_in_last_batch[i]

                # correct the values when drop_last=Fasle
                self._num_entries += 1
                self._num_samples = (self._num_entries - 1) * batch_size + self._samples_in_last_batch

        self._prefetch = min(prefetch, self._num_entries)
        self._prefetch_queue = queue.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._label_file = os.open(label_bin, os.O_RDONLY)
        self._dense_file = os.open(dense_bin, os.O_RDONLY)

        self._vocab_sizes = vocab_sizes
        self._category_file = [os.open(file, os.O_RDONLY) for file in category_bin]        
        self._category_type = [self._get_categorical_feature_type(size) for size in vocab_sizes]
        self._category_bytes = [np.dtype(dtype).itemsize for dtype in self._category_type]

    def _get_categorical_feature_type(self, size):
        types = (np.int8, np.int16, np.int32)
        for numpy_type in types:
            if size < np.iinfo(numpy_type).max:
                return numpy_type
        raise RuntimeError(f'Categorical feature of size {size} is too big for defined types')

    def _check_file(self, label_bin, dense_bin, category_bin, vocab_sizes):
        # num_samples represents the actual number of samples in the dataset
        num_samples = os.path.getsize(label_bin)
        if num_samples <= 0:
            raise RuntimeError("There must be at least one sample in %s"%label_bin)

        # check file size
        all_files = [(dense_bin, 13 * 2)]
        for i in range(26):
            dtype = self._get_categorical_feature_type(vocab_sizes[i])
            bytes_of_dtype = np.dtype(dtype).itemsize
            all_files.append((category_bin[i], bytes_of_dtype))

        for file, bytes_per_sample in all_files:
            file_size = os.path.getsize(file)
            if file_size % bytes_per_sample != 0:
                raise RuntimeError("The file size of %s should be an integer multiple of %d."%(file, bytes_per_sample))
            if (file_size // bytes_per_sample) != num_samples:
                raise RuntimeError("The number of samples in %s is not equeal to %s"%(dense_bin, label_bin))

    def __del__(self):
        for file in [self._label_file] + [self._dense_file] + self._category_file:
            if file is not None:
                os.close(file)

    def __len__(self):
        return self._num_entries

    def __getitem__(self, idx):
        if idx >= self._num_entries:
            raise IndexError()

        if self._prefetch <= 1:
            return self._get(idx)

        if idx == 0:
            for i in range(self._prefetch):
                self._prefetch_queue.put(self._executor.submit(self._get, (i)))

        if idx < (self._num_entries - self._prefetch):
            self._prefetch_queue.put(self._executor.submit(self._get, (idx + self._prefetch)))

        return self._prefetch_queue.get().result()

    def _get(self, idx):
        # calculate the offset & number of the samples to be read
        if not self._drop_last and idx == self._num_entries - 1:
            sample_offset = idx * (self._batch_size * self._global_size) + self._last_batch_offset[self._global_rank]
            batch = self._samples_in_last_batch
        else:
            sample_offset = idx * (self._batch_size * self._global_size) + (self._batch_size * self._global_rank)
            batch = self._batch_size

        # read the data from binary file
        label_raw_data = os.pread(self._label_file, 1 * batch, 1 * sample_offset)
        label = np.frombuffer(label_raw_data, dtype=np.bool_).reshape([batch, 1])

        dense_raw_data = os.pread(self._dense_file, 26 * batch, 26 * sample_offset)
        dense = np.frombuffer(dense_raw_data, dtype=np.float16).reshape([batch, 13])

        category = []
        for i in range(26):
            category_raw_data = os.pread(self._category_file[i], self._category_bytes[i] * batch, self._category_bytes[i] * sample_offset)
            category.append(np.frombuffer(category_raw_data, dtype=self._category_type[i]).reshape([batch, 1]).astype(np.int32))
        category = np.concatenate(category, axis=1)

        # convert numpy data to tensorflow data
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        dense = tf.convert_to_tensor(dense, dtype=tf.float32)
        category = tf.convert_to_tensor(category, dtype=tf.int64)

        return (dense, category), label


class SyntheticDataset:

    def __init__(self, batch_size, num_iterations, vocab_sizes, prefetch=1):
        self._batch_size = batch_size
        self._num_entries = num_iterations
        self._vocab_sizes = vocab_sizes

        self._prefetch = min(prefetch, self._num_entries)
        self._prefetch_queue = queue.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __len__(self):
        return self._num_entries

    def __getitem__(self, idx):
        if idx >= self._num_entries:
            raise IndexError()

        if self._prefetch <= 1:
            return self._get(idx)

        if idx == 0:
            for i in range(self._prefetch):
                self._prefetch_queue.put(self._executor.submit(self._get, (i)))

        if idx < (self._num_entries - self._prefetch):
            self._prefetch_queue.put(self._executor.submit(self._get, (idx + self._prefetch)))

        return self._prefetch_queue.get().result()

    def _get(self, idx):
        label = np.random.randint(0, 2, self._batch_size).reshape([-1, 1])
        dense = np.random.randint(0, 1024, self._batch_size*13).reshape([-1, 13])
        category = []
        for size in self._vocab_sizes:
            category.append(np.random.randint(0, size, self._batch_size).reshape([-1, 1]))
        category = np.concatenate(category, axis=1)

        # convert numpy data to tensorflow data
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        dense = tf.convert_to_tensor(dense, dtype=tf.float32)
        category = tf.convert_to_tensor(category, dtype=tf.int64)

        # preprocess
        dense = tf.math.log(dense + 3.0)

        return (dense, category), label

