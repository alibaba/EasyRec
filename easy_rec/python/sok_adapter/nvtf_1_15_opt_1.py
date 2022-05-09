
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training.optimizer import _TensorProcessor, _DenseResourceVariableProcessor, _RefVariableProcessor


def _get_processor(v):
    """The processor of v."""
    if context.executing_eagerly():
        if isinstance(v, ops.Tensor):
            return _TensorProcessor(v)
        else:
            return _DenseResourceVariableProcessor(v)
    if resource_variable_ops.is_resource_variable(v) and not v._in_graph_mode:  # pylint: disable=protected-access
        # True if and only if `v` was initialized eagerly.
        return _DenseResourceVariableProcessor(v)
    if v.op.type == "VarHandleOp":
        return _DenseResourceVariableProcessor(v)
    if isinstance(v, variables.Variable):
        return _RefVariableProcessor(v)
    if isinstance(v, ops.Tensor):
        return _TensorProcessor(v)
    raise NotImplementedError("Trying to optimize unsupported type ", v)


def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    # if distribute_ctx.has_strategy():
    # Handle DistributionStrategy case.
    # if distribute_ctx.in_cross_replica_context():
    #    raise RuntimeError("Use `_distributed_apply()` instead of "
    #                       "`apply_gradients()` in a cross-replica context.")
    #grads_and_vars = get_filtered_grad_fn(lambda: grads_and_vars)()
    # return distribute_ctx.get_replica_context().merge_call(
    #    self._distributed_apply, args=(grads_and_vars, global_step, name))

    name = name if name is not None else self.get_name()
    grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.

    def apply_fn():
        # No DistributionStrategy case.
        if not grads_and_vars:
            raise ValueError("No variables provided.")
        converted_grads_and_vars = []
        for g, v in grads_and_vars:
            if g is not None:
                try:
                    # Convert the grad to Tensor or IndexedSlices if necessary.
                    g = ops.convert_to_tensor_or_indexed_slices(g)
                except TypeError:
                    raise TypeError(
                        "Gradient must be convertible to a Tensor"
                        " or IndexedSlices, or None: %s" % g)
                if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
                    raise TypeError(
                        "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
            p = _get_processor(v)
            converted_grads_and_vars.append((g, v, p))

        converted_grads_and_vars = tuple(converted_grads_and_vars)
        var_list = [v for g, v, _ in converted_grads_and_vars if g is not None]
        if not var_list:
            raise ValueError("No gradients provided for any variable: %s." %
                             ([str(v) for _, v, _ in converted_grads_and_vars],))
        with ops.init_scope():
            self._create_slots(var_list)
        update_ops = []
        with ops.name_scope(name, self._name) as sname:
            self._prepare()
            for grad, var, processor in converted_grads_and_vars:
                if grad is None:
                    continue
                # We colocate all ops created in _apply_dense or _apply_sparse
                # on the same device as the variable.
                # TODO(apassos): figure out how to get the variable name here.
                if (context.executing_eagerly() or
                    isinstance(var, resource_variable_ops.BaseResourceVariable)
                        and not var._in_graph_mode):  # pylint: disable=protected-access
                    scope_name = ""
                else:
                    scope_name = var.op.name
                with ops.name_scope("update_" + scope_name), ops.colocate_with(var):
                    update_ops.append(processor.update_op(self, grad))
            if global_step is None:
                apply_updates = self._finish(update_ops, sname + '-apply')
            else:
                with ops.control_dependencies([self._finish(update_ops, "update")]):
                    with ops.colocate_with(global_step):
                        if isinstance(global_step, resource_variable_ops.ResourceVariable):
                            # TODO(apassos): the implicit read in assign_add is slow; consider
                            # making it less so.
                            apply_updates = resource_variable_ops.assign_add_variable_op(
                                global_step.handle,
                                ops.convert_to_tensor(1, dtype=global_step.dtype),
                                name=sname + '-apply')
                        else:
                            apply_updates = state_ops.assign_add(global_step, 1, name=sname + '-apply')

            if not context.executing_eagerly():
                if isinstance(apply_updates, ops.Tensor):
                    apply_updates = apply_updates.op
                train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
                if apply_updates not in train_op:
                    train_op.append(apply_updates)

            if not isinstance(apply_updates, ops.Operation) or apply_updates.type != 'NoOp':
                apply_updates = control_flow_ops.group([apply_updates])
            return apply_updates

    if self.doing_loss_scaling():
        grads = [g for g, _ in grads_and_vars]
        loss_scale_update_op, should_apply_grads = (self._loss_scale.update(grads))
        maybe_apply_op = smart_cond.smart_cond(should_apply_grads, apply_fn,
                                               control_flow_ops.no_op)
        return control_flow_ops.group(
            maybe_apply_op, loss_scale_update_op, name=name)
    else:
        return apply_fn()
