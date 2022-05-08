import logging

from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.training.optimizer import _get_processor
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import tf_utils


def _filter_grads(grads_and_vars):
    """Filter out iterable with grad equal to None."""
    grads_and_vars = tuple(grads_and_vars)
    if not grads_and_vars:
        return grads_and_vars
    filtered = []
    vars_with_empty_grads = []
    for grad, var in grads_and_vars:
        if grad is None:
            vars_with_empty_grads.append(var)
        else:
            filtered.append((grad, var))
    filtered = tuple(filtered)
    if not filtered:
        raise ValueError("No gradients provided for any variable: %s." %
                         ([v.name for _, v in grads_and_vars],))
    if vars_with_empty_grads:
        logging.warning(
            ("Gradients do not exist for variables %s when minimizing the loss."),
            ([v.name for v in vars_with_empty_grads]))
    return filtered


def apply_gradients(self, grads_and_vars, name=None):
    grads_and_vars = _filter_grads(grads_and_vars)
    var_list = [v for (_, v) in grads_and_vars]

    with backend.name_scope(self._name):
        # Create iteration if necessary.
        with ops.init_scope():
            _ = self.iterations
            self._create_hypers()
            self._create_slots(var_list)

        apply_state = self._prepare(var_list)
        # return distribute_ctx.get_replica_context().merge_call(
        #    functools.partial(self._distributed_apply, apply_state=apply_state),
        #    args=(grads_and_vars,),
        #    kwargs={"name": name})

        def apply_grad_to_update_var(var, grad):
            """Apply gradient to variable."""
            if isinstance(var, ops.Tensor):
                raise NotImplementedError("Trying to update a Tensor ", var)

            apply_kwargs = {}
            if isinstance(grad, ops.IndexedSlices):
                if var.constraint is not None:
                    raise RuntimeError(
                        "Cannot use a constraint function on a sparse variable.")
                if "apply_state" in self._sparse_apply_args:
                    apply_kwargs["apply_state"] = apply_state
                return self._resource_apply_sparse_duplicate_indices(
                    grad.values, var, grad.indices, **apply_kwargs)

            if "apply_state" in self._dense_apply_args:
                apply_kwargs["apply_state"] = apply_state
            update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
            if var.constraint is not None:
                with ops.control_dependencies([update_op]):
                    return var.assign(var.constraint(var))
            else:
                return update_op

        update_ops = []
        with backend.name_scope(name or self._name):
            for grad, var in grads_and_vars:
                scope_name = ("update" if ops.executing_eagerly_outside_functions() else
                              "update_" + var.op.name)
                # Colocate the update with variables to avoid unnecessary communication
                # delays. See b/136304694.
                with backend.name_scope(scope_name):
                    # distribution.extended.colocate_vars_with(var)
                    # update_ops.extend(
                    #    distribution.extended.update(
                    #        var, apply_grad_to_update_var, args=(grad,), group=False))
                    update_ops.append(apply_grad_to_update_var(var, grad))
            any_symbolic = any(isinstance(i, ops.Operation) or
                               tf_utils.is_symbolic_tensor(i) for i in update_ops)
            if not context.executing_eagerly() or any_symbolic:
                # If the current context is graph mode or any of the update ops are
                # symbolic then the step update should be carried out under a graph
                # context. (eager updates execute immediately)
                with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
                    with ops.control_dependencies(update_ops):
                        return self._iterations.assign_add(1).op

            return self._iterations.assign_add(1)


def sub_call(self, grads_and_vars, name, apply_state):
    # reduced_grads = distribution.extended.batch_reduce_to(
    #    ds_reduce_util.ReduceOp.SUM, grads_and_vars)
    #var_list = [v for _, v in grads_and_vars]
    #grads_and_vars = zip(reduced_grads, var_list)
    pass
