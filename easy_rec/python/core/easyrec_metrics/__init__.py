import logging
import os

import tensorflow as tf

from easy_rec.python.builders import loss_builder
from easy_rec.python.core import metrics as metrics_lib
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.loss_pb2 import LossType
from easy_rec.python.utils import pai_util

distribute_eval = os.environ.get('distribute_eval')
if distribute_eval == 'True':
  if pai_util.is_on_pai() or tf.__version__ <= '1.13':
    from easy_rec.python.core.easyrec_metrics import metrics_impl_pai as metrics_tf
  else:
    from easy_rec.python.core.easyrec_metrics import metrics_impl_tf as metrics_tf
else:
  from tensorflow import metrics as metrics_tf
