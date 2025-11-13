# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Define filters for restore."""

from abc import ABCMeta
from abc import abstractmethod
from enum import Enum


class Logical(Enum):
  AND = 1
  OR = 2


class Filter:
  __metaclass__ = ABCMeta

  def __init__(self):
    pass

  @abstractmethod
  def keep(self, var_name):
    """Keep the var or not.

    Args:
      var_name: input name of the var

    Returns:
      True if the var will be kept, else False
    """
    return True


class KeywordFilter(Filter):

  def __init__(self, pattern, exclusive=False):
    """Init KeywordFilter.

    Args:
      pattern: keyword to be matched
      exclusive: if True, var_name should include the pattern
          else, var_name should not include the pattern
    """
    self._pattern = pattern
    self._exclusive = exclusive

  def keep(self, var_name):
    if not self._exclusive:
      return self._pattern in var_name
    else:
      return self._pattern not in var_name


class CombineFilter(Filter):

  def __init__(self, filters, logical=Logical.AND):
    """Init CombineFilter.

    Args:
      filters: a set of filters to be combined
      logical: logical and/or combination of the filters
    """
    self._filters = filters
    self._logical = logical

  def keep(self, var_name):
    if self._logical == Logical.AND:
      for one_filter in self._filters:
        if not one_filter.keep(var_name):
          return False
      return True
    elif self._logical == Logical.OR:
      for one_filter in self._filters:
        if one_filter.keep(var_name):
          return True
      return False


class ScopeDrop:
  """For drop out scope prefix when restore variables from checkpoint."""

  def __init__(self, scope_name):
    self._scope_name = scope_name
    if len(self._scope_name) >= 0:
      if self._scope_name[-1] != '/':
        self._scope_name += '/'

  def update(self, var_name):
    return var_name.replace(self._scope_name, '')
