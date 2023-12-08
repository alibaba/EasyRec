import collections
import logging
import multiprocessing
import queue
import threading
import time
from multiprocessing import context

import numpy as np
import pandas as pd


def start_data_proc(task_index, task_num, num_proc, file_que, writers,
                    proc_start_sem, proc_stop_que, batch_size, label_fields,
                    effective_fields, reserve_fields, drop_remainder,
                    need_pack):
  mp_ctxt = multiprocessing.get_context('spawn')
  proc_arr = []
  for proc_id in range(num_proc):
    proc = mp_ctxt.Process(
        target=load_data_proc,
        args=(proc_id, file_que, writers[proc_id], proc_start_sem,
              proc_stop_que, batch_size, label_fields, effective_fields,
              reserve_fields, drop_remainder, task_index, task_num, need_pack),
        name='task_%d_data_proc_%d' % (task_index, proc_id))
    proc.daemon = True
    proc.start()
    proc_arr.append(proc)
  return proc_arr


def _should_stop(proc_stop_que):
  try:
    proc_stop_que.get(block=False)
    logging.info('data_proc stop signal received')
    proc_stop_que.close()
    return True
  except queue.Empty:
    return False
  except ValueError:
    return True
  except AssertionError:
    return True


def _add_to_que(data_dict, buffer):
  try:
    binary_data = context.reduction.ForkingPickler.dumps(data_dict)
    while len(buffer) >= buffer.maxlen:
      # logging.warning('send_que buffer is full')
      time.sleep(0.1)
    buffer.append(binary_data)
    return True
  except Exception as ex:
    logging.warning('add_to_que failed: %s' % str(ex))
    return False


def _get_one_file(file_que, proc_stop_que):
  while True:
    try:
      input_file = file_que.get(timeout=1)
      return input_file
    except queue.Empty:
      pass
  return None


def _pack_sparse_feas(data_dict, effective_fields):
  fea_val_arr = []
  fea_len_arr = []
  for fea_name in effective_fields:
    fea_len_arr.append(data_dict[fea_name][0])
    fea_val_arr.append(data_dict[fea_name][1])
    del data_dict[fea_name]
  fea_lens = np.concatenate(fea_len_arr, axis=0)
  fea_vals = np.concatenate(fea_val_arr, axis=0)
  data_dict['sparse_fea'] = (fea_lens, fea_vals)


def load_data_proc(proc_id, file_que, writer, proc_start_sem, proc_stop_que,
                   batch_size, label_fields, effective_fields, reserve_fields,
                   drop_remainder, task_index, task_num, need_pack):
  logging.info('data_proc[%d] wait start' % proc_id)
  proc_start_sem.acquire()
  logging.info('data_proc[%d] start' % proc_id)

  buffer = collections.deque(maxlen=32)

  is_good = True
  data_end = False

  def _send_func():
    total_send_ts = 0
    total_send_cnt = 0
    start_ts = time.time()
    while is_good:
      if len(buffer) == 0:
        if data_end:
          logging.info('data_proc[%d] send all data' % proc_id)
          break
        time.sleep(0.1)
        continue
      data = buffer.popleft()
      try:
        ts0 = time.time()
        writer.send_bytes(data)
        ts2 = time.time()
        total_send_ts += (ts2 - ts0)
        total_send_cnt += 1
        if total_send_cnt % 100 == 0:
          logging.info(
              ('data_proc[%d] send_time_stat: total_send_ts=%.3f ' +
               'total_send_cnt=%d total_ts=%d') %
              (proc_id, total_send_ts, total_send_cnt, time.time() - start_ts))
      except Exception as ex:
        logging.warning('send bytes exception: %s' % str(ex))
    logging.info(
        ('data_proc[%d] final send_time_stat: total_send_ts=%.3f ' +
         'total_send_cnt=%d total_ts=%.3f') %
        (proc_id, total_send_ts, total_send_cnt, time.time() - start_ts))

  send_thread = threading.Thread(target=_send_func)
  send_thread.start()

  all_fields = list(effective_fields)
  if label_fields is not None:
    all_fields = all_fields + label_fields
  if reserve_fields is not None:
    for tmp in reserve_fields:
      if tmp not in all_fields:
        all_fields.append(tmp)
  logging.info('data proc %d start, file_que.qsize=%d' %
               (proc_id, file_que.qsize()))
  num_files = 0
  part_data_dict = {}

  check_stop_ts = 0
  read_file_ts = 0
  parse_ts = 0
  parse_lbl_ts = 0
  parse_fea_ts = 0
  parse_fea_ts1 = 0
  parse_fea_ts2 = 0
  pack_ts = 0

  total_sample_num = 0
  total_batch_num = 0

  while is_good:
    ts0 = time.time()
    if _should_stop(proc_stop_que):
      is_good = False
      break
    input_file = _get_one_file(file_que, proc_stop_que)
    if input_file is None:
      break
    ts1 = time.time()
    check_stop_ts += (ts1 - ts0)
    num_files += 1
    input_data = pd.read_parquet(
        input_file, columns=all_fields, engine='pyarrow')
    data_len = len(input_data[all_fields[0]])
    total_sample_num += data_len

    batch_num = int(data_len / batch_size)
    res_num = data_len % batch_size

    ts2 = time.time()
    read_file_ts += (ts2 - ts1)
    # logging.info(
    #     'proc[%d] read file %s sample_num=%d batch_num=%d res_num=%d' %
    #     (proc_id, input_file, data_len, batch_num, res_num))
    # sub_batch_size = int(batch_size / task_num)
    sid = 0
    for batch_id in range(batch_num):
      eid = sid + batch_size
      data_dict = {}

      ts20 = time.time()
      if label_fields is not None and len(label_fields) > 0:
        for k in label_fields:
          data_dict[k] = np.array([x[0] for x in input_data[k][sid:eid]],
                                  dtype=np.int32)
      if reserve_fields is not None and len(reserve_fields) > 0:
        data_dict['reserve'] = {}
        for k in reserve_fields:
          np_dtype = type(input_data[k][0])
          if np_dtype == object:
            np_dtype = np.str
          data_dict['reserve'][k] = np.array(
              [x[0] for x in input_data[k][sid:eid]], dtype=np_dtype)
      ts21 = time.time()
      parse_lbl_ts += (ts21 - ts20)

      for k in effective_fields:
        val = input_data[k][sid:eid]
        # ts210 = time.time()
        all_lens = np.array([len(x) for x in val], dtype=np.int32)
        # ts211 = time.time()
        all_vals = np.concatenate(val.to_numpy(), axis=0)
        # ts212 = time.time()
        # parse_fea_ts1 += (ts211 - ts210)
        # parse_fea_ts2 += (ts212 - ts211)
        # assert np.sum(all_lens) == len(
        #     all_vals), 'len(all_vals)=%d np.sum(all_lens)=%d' % (
        #         len(all_vals), np.sum(all_lens))
        data_dict[k] = (all_lens, all_vals)

      ts22 = time.time()
      parse_fea_ts += (ts22 - ts21)

      if need_pack:
        _pack_sparse_feas(data_dict, effective_fields)

      ts23 = time.time()
      pack_ts += (ts23 - ts22)

      # logging.info('task_index=%d sid=%d eid=%d total_len=%d' % (task_index, sid, eid,
      #      len(data_dict['sparse_fea'][1])))
      if not _add_to_que(data_dict, buffer):
        logging.info('add to que failed')
        is_good = False
        break
      else:
        total_batch_num += 1
      sid += batch_size

    if res_num > 0 and is_good:
      data_dict = {}
      part_data_dict_n = {}

      ts20 = time.time()
      if label_fields is not None and len(label_fields) > 0:
        for k in label_fields:
          tmp_lbls = np.array([x[0] for x in input_data[k][sid:]],
                              dtype=np.float32)
          if part_data_dict is not None and k in part_data_dict:
            tmp_lbls = np.concatenate([part_data_dict[k], tmp_lbls], axis=0)
            if len(tmp_lbls) > batch_size:
              data_dict[k] = tmp_lbls[:batch_size]
              part_data_dict_n[k] = tmp_lbls[batch_size:]
            elif len(tmp_lbls) == batch_size:
              data_dict[k] = tmp_lbls
            else:
              part_data_dict_n[k] = tmp_lbls
          else:
            part_data_dict_n[k] = tmp_lbls

      if reserve_fields is not None and len(reserve_fields) > 0:
        data_dict['reserve'] = {}
        part_data_dict_n['reserve'] = {}
        for k in reserve_fields:
          np_dtype = type(input_data[k][0])
          if np_dtype == object:
            np_dtype = np.str
          tmp_r = np.array([x[0] for x in input_data[k][sid:]], dtype=np_dtype)
          if part_data_dict is not None and 'reserve' in part_data_dict and \
             k in part_data_dict['reserve']:
            tmp_r = np.concatenate([part_data_dict['reserve'][k], tmp_r],
                                   axis=0)
            if len(tmp_r) > batch_size:
              data_dict['reserve'][k] = tmp_r[:batch_size]
              part_data_dict_n['reserve'][k] = tmp_r[batch_size:]
            elif len(tmp_r) == batch_size:
              data_dict['reserve'][k] = tmp_r
            else:
              part_data_dict_n['reserve'][k] = tmp_r
          else:
            part_data_dict_n['reserve'][k] = tmp_r
      ts21 = time.time()
      parse_lbl_ts += (ts21 - ts20)

      for k in effective_fields:
        val = input_data[k][sid:]
        all_lens = np.array([len(x) for x in val], dtype=np.int32)
        all_vals = np.concatenate(val.to_numpy())
        if part_data_dict is not None and k in part_data_dict:
          tmp_lens = np.concatenate([part_data_dict[k][0], all_lens], axis=0)
          tmp_vals = np.concatenate([part_data_dict[k][1], all_vals], axis=0)
          if len(tmp_lens) > batch_size:
            tmp_res_lens = tmp_lens[batch_size:]
            tmp_lens = tmp_lens[:batch_size]
            tmp_num_elems = np.sum(tmp_lens)
            tmp_res_vals = tmp_vals[tmp_num_elems:]
            tmp_vals = tmp_vals[:tmp_num_elems]
            part_data_dict_n[k] = (tmp_res_lens, tmp_res_vals)
            data_dict[k] = (tmp_lens, tmp_vals)
          elif len(tmp_lens) == batch_size:
            data_dict[k] = (tmp_lens, tmp_vals)
          else:
            part_data_dict_n[k] = (tmp_lens, tmp_vals)
        else:
          part_data_dict_n[k] = (all_lens, all_vals)

      ts22 = time.time()
      parse_fea_ts += (ts22 - ts21)

      if effective_fields[0] in data_dict:
        if need_pack:
          _pack_sparse_feas(data_dict, effective_fields)
        if not _add_to_que(data_dict, buffer):
          logging.info('add to que failed')
          is_good = False
          break
        else:
          total_batch_num += 1
      ts23 = time.time()
      pack_ts += (ts23 - ts22)

      part_data_dict = part_data_dict_n
    ts3 = time.time()
    parse_ts += (ts3 - ts2)
  if len(part_data_dict) > 0 and is_good:
    if not drop_remainder:
      if need_pack:
        _pack_sparse_feas(part_data_dict, effective_fields)
      _add_to_que(part_data_dict, buffer)
      total_batch_num += 1
    else:
      logging.warning('drop remain %d samples as drop_remainder is set' %
                      len(part_data_dict[effective_fields[0]]))
  if is_good:
    is_good = _add_to_que(None, buffer)

  data_end = True
  send_thread.join()

  logging.info((
      'data_proc[%d], is_good=%s, check_stop_ts=%.3f, ' +
      'read_file_ts=%.3f, parse_ts=%.3f, parse_lbl_ts=%.3f, parse_fea_ts=%.3f, '
      + 'parse_fea_ts1=%.3f, parse_fea_ts2=%.3f, pack_ts=%.3f') % (
          proc_id,
          is_good,
          check_stop_ts,
          read_file_ts,
          parse_ts,  # yapf:skip
          parse_lbl_ts,
          parse_fea_ts,
          parse_fea_ts1,
          parse_fea_ts2,  # yapf:skip
          pack_ts))
  writer.close()

  while not is_good:
    try:
      if file_que.get(timeout=1) is None:
        break
    except queue.Empty:
      pass
  file_que.close()
  logging.info(
      'data_proc[%d] done: file_num=%d total_sample_num=%d total_batch_num=%d' %
      (proc_id, num_files, total_sample_num, total_batch_num))
