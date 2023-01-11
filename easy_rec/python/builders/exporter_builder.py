# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

# when version of tensorflow > 1.8 strip_default_attrs set true will cause
# saved_model inference core, such as:
#   [libprotobuf FATAL external/protobuf_archive/src/google/protobuf/map.h:1058]
#    CHECK failed: it != end(): key not found: new_axis_mask
# so temporarily modify strip_default_attrs of _SavedModelExporter in
# tf.estimator.exporter to false by default

import logging

from easy_rec.python.compat import exporter
from easy_rec.python.utils import config_util


def build(exporter_type, export_config, export_input_fn):
  exporter_types = [
      x.strip() for x in exporter_type.split(',') if x.strip() != ''
  ]
  exporters = []
  for tmp_type in exporter_types:
    if tmp_type == 'final':
      exporters.append(
          exporter.FinalExporter(
              name='final', serving_input_receiver_fn=export_input_fn))
    elif tmp_type == 'latest':
      exporters.append(
          exporter.LatestExporter(
              name='latest',
              serving_input_receiver_fn=export_input_fn,
              exports_to_keep=export_config.exports_to_keep))
    elif tmp_type == 'large':
      extra_params = config_util.parse_oss_params(export_config.oss_params)
      exporters.append(
          exporter.LargeExporter(
              name='large',
              serving_input_receiver_fn=export_input_fn,
              extra_params=extra_params,
              exports_to_keep=export_config.exports_to_keep))
    elif tmp_type == 'best':
      logging.info(
          'will use BestExporter, metric is %s, the bigger the better: %d' %
          (export_config.best_exporter_metric, export_config.metric_bigger))

      def _metric_cmp_fn(best_eval_result, current_eval_result):
        logging.info('metric: best = %s current = %s' %
                     (str(best_eval_result), str(current_eval_result)))
        if export_config.metric_bigger:
          return (best_eval_result[export_config.best_exporter_metric] <
                  current_eval_result[export_config.best_exporter_metric])
        else:
          return (best_eval_result[export_config.best_exporter_metric] >
                  current_eval_result[export_config.best_exporter_metric])

      exporters.append(
          exporter.BestExporter(
              name='best',
              serving_input_receiver_fn=export_input_fn,
              compare_fn=_metric_cmp_fn,
              exports_to_keep=export_config.exports_to_keep))
    elif tmp_type == 'none':
      continue
    else:
      raise ValueError('Unknown exporter type %s' % tmp_type)

  return exporters
