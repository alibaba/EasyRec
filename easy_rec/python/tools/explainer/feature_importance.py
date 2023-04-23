from __future__ import print_function
from easy_rec.python.tools.explainer.explainer import run
import tensorflow as tf
flags = tf.app.flags

flags.DEFINE_string('saved_model_dir', '', 'directory where saved_model.pb exists')
flags.DEFINE_string('explain_tables', '', 'tables used for explaination')
flags.DEFINE_string('background_table', '', 'tables used for expected value')
flags.DEFINE_string('tables', '', 'tables passed by pai command')
flags.DEFINE_string('outputs', '', 'output tables')
flags.DEFINE_string(
    'selected_cols', '',
    'columns to keep from input table,  they are separated with ,')
flags.DEFINE_string(
    'reserved_cols', '',
    'columns to keep from input table,  they are separated with ,')
flags.DEFINE_string(
    'output_cols', None,
    'output columns, such as: score float. multiple columns are separated by ,')
flags.DEFINE_integer('batch_size', 1024, 'predict batch size')
flags.DEFINE_string('worker_hosts', '', 'Comma-separated list of hostname:port pairs')
flags.DEFINE_integer('task_index', 0, 'Index of task within the job')

FLAGS = flags.FLAGS


def main(_):
  for k in FLAGS:
    if k in ('h', 'help', 'helpshort', 'helpfull'):
      continue
    print("%s=%s" % (k, FLAGS[k].value))

  # worker_count = len(FLAGS.worker_hosts.split(','))
  # e = create_explainer(FLAGS.saved_model_dir)
  #
  # output_names = e.input_names
  # print("feature_names:", output_names)
  # print("feature_num:", len(output_names))
  # e.feature_importance(FLAGS.explain_tables if FLAGS.explain_tables else FLAGS.tables,
  #                      FLAGS.outputs,
  #                      reserved_cols=FLAGS.reserved_cols,
  #                      output_cols=FLAGS.output_cols,
  #                      batch_size=FLAGS.batch_size,
  #                      slice_id=FLAGS.task_index,
  #                      slice_num=worker_count)
  run(FLAGS)


if __name__ == '__main__':
  tf.app.run(main=main)
