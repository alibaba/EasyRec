# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.


def copy_obj(proto_obj):
  """Make a copy of proto_obj so that later modifications of tmp_obj will have no impact on proto_obj.

  Args:
    proto_obj: a protobuf message
  Return:
    a copy of proto_obj
  """
  tmp_obj = type(proto_obj)()
  tmp_obj.CopyFrom(proto_obj)
  return tmp_obj
