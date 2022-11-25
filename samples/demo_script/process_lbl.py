import numpy as np


def remap_lbl(labels):
  res = np.where(labels < 5, 0, 1)
  return res
