import traceback
import logging

import tensorflow as tf

class ErrorImportedSOK(object):
    def __getattr__(self, _):
        raise RuntimeError("SOK init and importing failed, check your setup process and env")

try:
    logging.info("Try importing sparse_operation_kit ...")
    import sparse_operation_kit
    # test if has Init function
    sparse_operation_kit.Init
except Exception as e:
    logging.error(traceback.format_exc())
    logging.info("sparse_operation_kit module importing failed. Continue running the program,"
          "but anything related with SOK will not work properly")
    sparse_operation_kit = ErrorImportedSOK()
