import concurrent
import concurrent.futures
import os
import queue
import time

import numpy as np
import tensorflow as tf

# import horovod.tensorflow as hvd


class BinaryDataset:

  def __init__(
      self,
      label_bin,
      dense_bin,
      category_bin,
      batch_size=1,
      drop_last=False,
      prefetch=1,
      global_rank=0,
      global_size=1,
  ):
    file_size = os.path.getsize(label_bin)
    if file_size % 4 != 0:
      raise RuntimeError(
          'The file size of {} should be an integer multiple of 4.'.format(
              label_bin))
    num_samples = file_size // 4
    assert (os.path.getsize(dense_bin) // 52 == num_samples)
    assert (os.path.getsize(category_bin) // 104 == num_samples)

    if global_rank == (global_size - 1):
      self._num_samples = num_samples - (num_samples //
                                         global_size) * global_rank
    else:
      self._num_samples = num_samples // global_size

    self._bytes_offset_label = num_samples // global_size * 4 * global_rank
    self._bytes_offset_dense = num_samples // global_size * 52 * global_rank
    self._bytes_offset_category = num_samples // global_size * 104 * global_rank

    self._num_entries = self._num_samples // batch_size
    if not drop_last and (self._num_samples % batch_size != 0):
      self._num_entries += 1

    self._batch_size = batch_size
    self._drop_last = drop_last

    self._prefetch = min(prefetch, self._num_entries)
    self._prefetch_queue = queue.Queue()
    self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    self._label_file = os.open(label_bin, os.O_RDONLY)
    self._dense_file = os.open(dense_bin, os.O_RDONLY)
    self._category_file = os.open(category_bin, os.O_RDONLY)

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
      self._prefetch_queue.put(
          self._executor.submit(self._get, (idx + self._prefetch)))

    return self._prefetch_queue.get().result()

  def _get(self, idx):
    batch = self._batch_size
    if (idx == self._num_entries - 1) and not self._drop_last and (
        self._num_samples % self._batch_size != 0):
      batch = self._num_samples % self._batch_size

    label_raw_data = os.pread(
        self._label_file, 4 * batch,
        self._bytes_offset_label + idx * self._batch_size * 4)
    label = np.frombuffer(label_raw_data, dtype=np.int32).reshape([batch, 1])

    dense_raw_data = os.pread(
        self._dense_file, 52 * batch,
        self._bytes_offset_dense + idx * self._batch_size * 52)
    dense = np.frombuffer(dense_raw_data, dtype=np.float32).reshape([batch, 13])
    dense = np.log(dense + 3, dtype=np.float32)

    category_raw_data = os.pread(
        self._category_file, 104 * batch,
        self._bytes_offset_category + idx * self._batch_size * 104)
    category = np.frombuffer(
        category_raw_data, dtype=np.float32).reshape([batch, 26])

    return dense, category, label


class BinaryDataset2:

  def __init__(
      self,
      label_bin,
      dense_bin,
      category_bin,
      batch_size=1,
      drop_last=False,
      prefetch=1,
      global_rank=0,
      global_size=1,
  ):

    file_size = os.path.getsize(label_bin)
    if file_size % 4 != 0:
      raise RuntimeError(
          'The file size of {} should be an integer multiple of 4.'.format(
              label_bin))
    num_samples = file_size // 4
    assert (os.path.getsize(dense_bin) // 52 == num_samples)
    assert (os.path.getsize(category_bin) // 104 == num_samples)

    self._num_entries = num_samples // (batch_size * global_size)
    self._num_samples = self._num_entries * batch_size
    assert ((num_samples -
             self._num_samples * global_size) == (num_samples %
                                                  (batch_size * global_size)))

    self._batch_size = batch_size
    self._drop_last = drop_last
    self._global_rank = global_rank
    self._global_size = global_size

    self._prefetch = min(prefetch, self._num_entries)
    self._prefetch_queue = queue.Queue()
    self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    self._label_file = os.open(label_bin, os.O_RDONLY)
    self._dense_file = os.open(dense_bin, os.O_RDONLY)
    self._category_file = os.open(category_bin, os.O_RDONLY)

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
      self._prefetch_queue.put(
          self._executor.submit(self._get, (idx + self._prefetch)))

    return self._prefetch_queue.get().result()

  def _get(self, idx):
    sample_offset = idx * self._batch_size * self._global_size + self._global_rank * self._batch_size

    label_raw_data = os.pread(self._label_file, 4 * self._batch_size,
                              4 * sample_offset)
    label = np.frombuffer(
        label_raw_data, dtype=np.int32).reshape([self._batch_size, 1])

    dense_raw_data = os.pread(self._dense_file, 52 * self._batch_size,
                              52 * sample_offset)
    dense = np.frombuffer(
        dense_raw_data, dtype=np.float32).reshape([self._batch_size, 13])
    dense = np.log(dense + 3, dtype=np.float32)

    category_raw_data = os.pread(self._category_file, 104 * self._batch_size,
                                 104 * sample_offset)
    category = np.frombuffer(
        category_raw_data, dtype=np.float32).reshape([self._batch_size, 26])

    return dense, category, label


if __name__ == '__main__':
  global_batch_size = 1024
  dataset_dir = './dataset/'
  test_dataset = BinaryDataset(
      dataset_dir + 'label.bin',
      dataset_dir + 'dense.bin',
      dataset_dir + 'category.bin',
      batch_size=global_batch_size,
      drop_last=False,
      prefetch=10,
      global_rank=0,
      global_size=1,
  )

  for step, (dense, category, labels) in enumerate(test_dataset):
    if (step % 100 == 0):
      print(step, 0)
    if step == 0:
      print('warmup over!')
      start_time = time.time()
    if step == 1000:
      print('1000 steps ==', time.time() - start_time, 0)
