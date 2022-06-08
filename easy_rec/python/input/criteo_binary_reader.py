# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import concurrent
import concurrent.futures
import glob
import logging
import os
import queue
import time

import numpy as np


class BinaryDataset:

  def __init__(
      self,
      label_bins,
      dense_bins,
      category_bins,
      batch_size=1,
      drop_last=False,
      prefetch=1,
      global_rank=0,
      global_size=1,
  ):
    total_sample_num = 0
    self._sample_num_arr = []
    for label_bin in label_bins:
      sample_num = os.path.getsize(label_bin) // 4
      total_sample_num += sample_num
      self._sample_num_arr.append(sample_num)
    logging.info('total number samples = %d' % total_sample_num)
    self._total_sample_num = total_sample_num

    self._batch_size = batch_size

    self._compute_global_start_pos(total_sample_num, batch_size, global_rank,
                                   global_size, drop_last)

    self._label_file_arr = [None for _ in self._sample_num_arr]
    self._dense_file_arr = [None for _ in self._sample_num_arr]
    self._category_file_arr = [None for _ in self._sample_num_arr]

    for tmp_file_id in range(self._start_file_id, self._end_file_id + 1):
      self._label_file_arr[tmp_file_id] = os.open(label_bins[tmp_file_id],
                                                  os.O_RDONLY)
      self._dense_file_arr[tmp_file_id] = os.open(dense_bins[tmp_file_id],
                                                  os.O_RDONLY)
      self._category_file_arr[tmp_file_id] = os.open(category_bins[tmp_file_id],
                                                     os.O_RDONLY)

    self._prefetch = min(prefetch, self._num_entries)
    self._prefetch_queue = queue.Queue()
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=self._prefetch)

    self._os_close_func = os.close

  def _compute_global_start_pos(self, total_sample_num, batch_size, global_rank,
                                global_size, drop_last):
    # ensure all workers have the same number of samples
    avg_sample_num = (total_sample_num // global_size)
    res_num = (total_sample_num % global_size)
    self._num_samples = avg_sample_num
    if res_num > 0:
      self._num_samples += 1
      if global_rank < res_num:
        global_start_pos = (avg_sample_num + 1) * global_rank
      else:
        global_start_pos = avg_sample_num * global_rank + res_num - 1
    else:
      global_start_pos = avg_sample_num * global_rank
    # global_end_pos = global_start_pos + self._num_samples

    self._num_entries = self._num_samples // batch_size
    self._last_batch_size = batch_size
    if not drop_last and (self._num_samples % batch_size != 0):
      self._num_entries += 1
      self._last_batch_size = self._num_samples % batch_size
    logging.info('num_batches = %d num_samples = %d' %
                 (self._num_entries, self._num_samples))

    start_file_id = 0
    curr_pos = 0
    while curr_pos + self._sample_num_arr[start_file_id] <= global_start_pos:
      start_file_id += 1
      curr_pos += self._sample_num_arr[start_file_id]
    self._start_file_id = start_file_id
    self._start_file_pos = global_start_pos - curr_pos

    logging.info('start_file_id = %d start_file_pos = %d' %
                 (start_file_id, self._start_file_pos))

    # find the start of each batch
    self._start_pos_arr = np.zeros([self._num_entries, 2], dtype=np.uint32)
    batch_id = 0
    tmp_start_pos = self._start_file_pos
    while batch_id < self._num_entries:
      self._start_pos_arr[batch_id] = (start_file_id, tmp_start_pos)
      batch_id += 1
      # the last batch
      if batch_id == self._num_entries:
        tmp_start_pos += self._last_batch_size
        while start_file_id < len(
            self._sample_num_arr
        ) and tmp_start_pos > self._sample_num_arr[start_file_id]:
          tmp_start_pos -= self._sample_num_arr[start_file_id]
          start_file_id += 1
      else:
        tmp_start_pos += batch_size
        while start_file_id < len(
            self._sample_num_arr
        ) and tmp_start_pos >= self._sample_num_arr[start_file_id]:
          tmp_start_pos -= self._sample_num_arr[start_file_id]
          start_file_id += 1

    self._end_file_id = start_file_id
    self._end_file_pos = tmp_start_pos

    logging.info('end_file_id = %d end_file_pos = %d' %
                 (self._end_file_id, self._end_file_pos))

  def __del__(self):
    for f in self._label_file_arr:
      if f is not None:
        self._os_close_func(f)
    for f in self._dense_file_arr:
      if f is not None:
        self._os_close_func(f)
    for f in self._category_file_arr:
      if f is not None:
        self._os_close_func(f)

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
    curr_file_id = self._start_pos_arr[idx][0]
    start_read_pos = self._start_pos_arr[idx][1]

    end_read_pos = start_read_pos + self._batch_size
    total_read_num = 0

    label_read_arr = []
    dense_read_arr = []
    cate_read_arr = []
    while total_read_num < self._batch_size and curr_file_id < len(
        self._sample_num_arr):
      tmp_read_num = min(end_read_pos,
                         self._sample_num_arr[curr_file_id]) - start_read_pos

      label_raw_data = os.pread(self._label_file_arr[curr_file_id],
                                4 * tmp_read_num, start_read_pos * 4)
      tmp_lbl_np = np.frombuffer(
          label_raw_data, dtype=np.int32).reshape([tmp_read_num, 1])
      label_read_arr.append(tmp_lbl_np)

      dense_raw_data = os.pread(self._dense_file_arr[curr_file_id],
                                52 * tmp_read_num, start_read_pos * 52)
      part_dense_np = np.frombuffer(
          dense_raw_data, dtype=np.float32).reshape([tmp_read_num, 13])
      # part_dense_np = np.log(part_dense_np + 3, dtype=np.float32)
      dense_read_arr.append(part_dense_np)

      category_raw_data = os.pread(self._category_file_arr[curr_file_id],
                                   104 * tmp_read_num, start_read_pos * 104)
      part_cate_np = np.frombuffer(
          category_raw_data, dtype=np.uint32).reshape([tmp_read_num, 26])
      cate_read_arr.append(part_cate_np)

      curr_file_id += 1
      start_read_pos = 0
      total_read_num += tmp_read_num

    if len(label_read_arr) == 1:
      label = label_read_arr[0]
    else:
      label = np.concatenate(label_read_arr, axis=0)

    if len(cate_read_arr) == 1:
      category = cate_read_arr[0]
    else:
      category = np.concatenate(cate_read_arr, axis=0)

    if len(dense_read_arr) == 1:
      dense = dense_read_arr[0]
    else:
      dense = np.concatenate(dense_read_arr, axis=0)

    return dense, category, label


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
  parser.add_argument(
      '--dataset_dir', type=str, default='./', help='dataset_dir')
  parser.add_argument('--task_num', type=int, default=1, help='task number')
  parser.add_argument('--task_index', type=int, default=0, help='task index')
  parser.add_argument(
      '--prefetch_size', type=int, default=10, help='prefetch size')
  args = parser.parse_args()

  batch_size = args.batch_size
  dataset_dir = args.dataset_dir
  logging.info('batch_size = %d' % batch_size)
  logging.info('dataset_dir = %s' % dataset_dir)

  label_files = glob.glob(os.path.join(dataset_dir, '*_label.bin'))
  dense_files = glob.glob(os.path.join(dataset_dir, '*_dense.bin'))
  category_files = glob.glob(os.path.join(dataset_dir, '*_category.bin'))

  label_files.sort()
  dense_files.sort()
  category_files.sort()

  test_dataset = BinaryDataset(
      label_files,
      dense_files,
      category_files,
      batch_size=batch_size,
      drop_last=False,
      prefetch=args.prefetch_size,
      global_rank=args.task_index,
      global_size=args.task_num,
  )

  for step, (dense, category, labels) in enumerate(test_dataset):
    # if (step % 100 == 0):
    #   print(step, dense.shape, category.shape, labels.shape)
    if step == 0:
      logging.info('warmup over!')
      start_time = time.time()
    if step == 1000:
      logging.info('1000 steps time = %.3f' % (time.time() - start_time))
  logging.info('total_steps = %d total_time = %.3f' %
               (step + 1, time.time() - start_time))
  logging.info(
      'final step[%d] dense_shape=%s category_shape=%s labels_shape=%s' %
      (step, dense.shape, category.shape, labels.shape))
