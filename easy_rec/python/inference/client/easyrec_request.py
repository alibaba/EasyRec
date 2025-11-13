# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from eas_prediction.request import Request

from easy_rec.python.protos.predict_pb2 import PBRequest
from easy_rec.python.protos.predict_pb2 import PBResponse

# from eas_prediction.request import Response


class EasyrecRequest(Request):
  """Request for tensorflow services whose input data is in format of protobuf.

  This class privide methods to fill generate PBRequest and parse PBResponse.
  """

  def __init__(self, signature_name=None):
    self.request_data = PBRequest()
    self.signature_name = signature_name

  def __str__(self):
    return self.request_data

  def set_signature_name(self, singature_name):
    """Set the signature name of the model.

    Args:
      singature_name: signature name of the model
    """
    self.signature_name = singature_name

  def add_feed(self, data, dbg_lvl=0):
    if not isinstance(data, PBRequest):
      self.request_data.ParseFromString(data)
    else:
      self.request_data = data
    self.request_data.debug_level = dbg_lvl

  def add_user_fea_flt(self, k, v):
    self.request_data.user_features[k].float_feature = float(v)

  def add_user_fea_s(self, k, v):
    self.request_data.user_features[k].string_feature = str(v)

  def set_faiss_neigh_num(self, neigh_num):
    self.request_data.faiss_neigh_num = neigh_num

  def keep_one_item_ids(self):
    item_id = self.request_data.item_ids[0]
    self.request_data.ClearField('item_ids')
    self.request_data.item_ids.extend([item_id])

  def to_string(self):
    """Serialize the request to string for transmission.

    Returns:
      the request data in format of string
    """
    return self.request_data.SerializeToString()

  def parse_response(self, response_data):
    """Parse the given response data in string format to the related TFResponse object.

    Args:
      response_data: the service response data in string format

    Returns:
      the TFResponse object related the request
    """
    self.response = PBResponse()
    self.response.ParseFromString(response_data)
    return self.response
