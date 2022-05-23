import traceback
import logging
import inspect
import types

import tensorflow
from easy_rec.python.sok_adapter import nvtf_1_15_opt_2


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
    sok_import_sucess = False
else:
    sok_import_sucess = True

if tensorflow.__version__ == '1.15.5':
    custom_apply_gradient = nvtf_1_15_opt_2.apply_gradients
else:
    raise RuntimeError("Not able to import custom_apply_gradient")
def modify_apply_gradients(optimizer):
    logging.info("Modifying {} 's apply_gradients".format(str(optimizer)))
    optimizer.apply_gradients = types.MethodType(custom_apply_gradient, optimizer)
    return optimizer