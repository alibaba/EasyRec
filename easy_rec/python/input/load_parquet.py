import logging
import multiprocessing
import queue

import numpy as np
import pandas as pd


def start_data_proc(task_index, task_num, num_proc, file_que, data_que,
                    proc_start_que, proc_stop_que, batch_size, label_fields,
                    effective_fields, reserve_fields, drop_remainder):
  mp_ctxt = multiprocessing.get_context('spawn')
  proc_arr = []
  for proc_id in range(num_proc):
    proc = mp_ctxt.Process(
        target=load_data_proc,
        args=(proc_id, file_que, data_que, proc_start_que, proc_stop_que,
              batch_size, label_fields, effective_fields, reserve_fields,
              drop_remainder, task_index, task_num),
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


def _add_to_que(data_dict, data_que, proc_stop_que):
  while True:
    try:
      data_que.put(data_dict, timeout=5)
      return True
    except queue.Full:
      logging.warning('data_que is full')
      if _should_stop(proc_stop_que):
        return False
    except ValueError:
      logging.warning('data_que is closed')
      return False
    except AssertionError:
      logging.warning('data_que is closed')
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


def load_data_proc(proc_id, file_que, data_que, proc_start_que, proc_stop_que,
                   batch_size, label_fields, effective_fields, reserve_fields,
                   drop_remainder, task_index, task_num):
  logging.info('data proc %d start, proc_start_que=%s' %
               (proc_id, proc_start_que.qsize()))
  proc_start_que.get()
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

  is_good = True
  while is_good:
    if _should_stop(proc_stop_que):
      is_good = False
      break
    input_file = _get_one_file(file_que, proc_stop_que)
    if input_file is None:
      break
    num_files += 1
    input_data = pd.read_parquet(input_file, columns=all_fields)
    data_len = len(input_data[all_fields[0]])
    batch_num = int(data_len / batch_size)
    res_num = data_len % batch_size
    # logging.info(
    #     'proc[%d] read file %s sample_num=%d batch_num=%d res_num=%d' %
    #     (proc_id, input_file, data_len, batch_num, res_num))
    # sub_batch_size = int(batch_size / task_num)
    sid = 0
    for batch_id in range(batch_num):
      eid = sid + batch_size
      data_dict = {}

      # sid_stub = sid
      # sid = sid + sub_batch_size * task_index
      # eid = sid + sub_batch_size

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

      for k in effective_fields:
        val = input_data[k][sid:eid]
        all_lens = np.array([len(x) for x in val], dtype=np.int32)
        all_vals = np.concatenate(val.to_numpy())
        assert np.sum(all_lens) == len(
            all_vals), 'len(all_vals)=%d np.sum(all_lens)=%d' % (
                len(all_vals), np.sum(all_lens))
        data_dict[k] = (all_lens, all_vals)

      _pack_sparse_feas(data_dict, effective_fields)
      # logging.info('task_index=%d sid=%d eid=%d total_len=%d' % (task_index, sid, eid,
      #      len(data_dict['sparse_fea'][1])))
      if not _add_to_que(data_dict, data_que, proc_stop_que):
        logging.info('add to que failed')
        is_good = False
        break
      sid += batch_size
    #   sid = batch_size + sid_stub
    # return

    if res_num > 0 and is_good:
      data_dict = {}
      part_data_dict_n = {}

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
      if 'sparse_fea' in data_dict:
        _pack_sparse_feas(data_dict, effective_fields)
        if not _add_to_que(data_dict, data_que, proc_stop_que):
          logging.info('add to que failed')
          is_good = False
          break

      part_data_dict = part_data_dict_n
  if len(part_data_dict) > 0 and is_good:
    if not drop_remainder:
      _pack_sparse_feas(part_data_dict, effective_fields)
      _add_to_que(part_data_dict, data_que, proc_stop_que)
    else:
      logging.warning('drop remain %d samples as drop_remainder is set' %
                      len(part_data_dict[effective_fields[0]]))
  if is_good:
    is_good = _add_to_que(None, data_que, proc_stop_que)
  logging.info('data_proc_id=%d, is_good = %s' % (proc_id, is_good))
  data_que.close(wait_send_finish=is_good)

  while not is_good:
    try:
      if file_que.get(timeout=1) is None:
        break
    except queue.Empty:
      pass
  file_que.close()
  logging.info('data proc %d done, file_num=%d' % (proc_id, num_files))
