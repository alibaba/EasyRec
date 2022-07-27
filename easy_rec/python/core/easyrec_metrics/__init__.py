import logging
import os

import tensorflow as tf

from easy_rec.python.utils import pai_util

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

distribute_eval = os.environ.get('distribute_eval')
logging.info('distribute_eval = {}'.format(distribute_eval))
if distribute_eval == 'True':
  if pai_util.is_on_pai() or tf.__version__ <= '1.13':
    logging.info('Will use distribute pai_tf metrics impl')
    from easy_rec.python.core.easyrec_metrics import distribute_metrics_impl_pai as metrics_tf
  else:
    logging.info('Will use distribute tf metrics impl')
    from easy_rec.python.core.easyrec_metrics import distribute_metrics_impl_tf as metrics_tf
else:
  if tf.__version__ >= '2.0':
    from tensorflow.compat.v1 import metrics as metrics_tf
  else:
    from tensorflow import metrics as metrics_tf
