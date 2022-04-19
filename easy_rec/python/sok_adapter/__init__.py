import traceback
import logging
import inspect

import tensorflow


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

if sok_import_sucess:
    logging.info("Try modifying tensorflow.train.Optimizer._distributed_apply.")

    original_distributed_apply = tensorflow.train.Optimizer._distributed_apply
    spec = inspect.getfullargspec(original_distributed_apply)
    try:
        grads_and_vars_idx_in_args = spec.args.index('grads_and_vars')
    except ValueError:
        logging.error("Can not found arg 'grads_and_vars' in tensorflow.train.Optimizer._distributed_apply."
                      " Modifying failed")
    else:
        def sok_split_vars_wrapper(*args, **kwargs):
            grads_and_vars = args[grads_and_vars_idx_in_args]
            logging.info("Splitting grads_and_vars in sok_split_vars_wrapper")
            return original_distributed_apply(*args, **kwargs)
        tensorflow.train.Optimizer._distributed_apply = sok_split_vars_wrapper
        logging.info("Wrapped it with sok_split_vars_wrapper")

