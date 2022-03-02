from tensorflow.python.framework import ops


class GraphKeys(ops.GraphKeys):
  # For rank service
  RANK_SERVICE_FG_CONF = '__rank_service_fg_conf'
  RANK_SERVICE_INPUT = '__rank_service_input'
  RANK_SERVICE_OUTPUT = '__rank_service_output'
  RANK_SERVICE_EMBEDDING = '__rank_service_embedding'
  RANK_SERVICE_INPUT_SRC = '__rank_service_input_src'
  RANK_SERVICE_REPLACE_OP = '__rank_service_replace'
  RANK_SERVICE_SHAPE_OPT_FLAG = '__rank_service_shape_opt_flag'
  # For compatition between RTP and EasyRec
  RANK_SERVICE_FEATURE_NODE = '__rank_service_feature_node'
