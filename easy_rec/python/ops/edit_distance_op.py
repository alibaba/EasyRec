import logging
import os

import tensorflow as tf
from tensorflow.python.util.tf_export import tf_export

import easy_rec

custom_op_path = os.path.join(easy_rec.ops_dir, 'libedit_distance.so')
print('custom op path: %s' % custom_op_path)

try:
  custom_ops = tf.load_op_library(custom_op_path)
  logging.info('load edit_distance op from %s succeed' % custom_op_path)
except Exception as ex:
  print('custom op path: %s' % custom_op_path)
  logging.warning('load edit_distance op failed: %s' % str(ex))
  custom_ops = None


@tf_export('edit_distance')
def edit_distance(input1, input2):
  return custom_ops.edit_distance_op(input1, input2)
