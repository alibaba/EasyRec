# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import gradients_impl as tf_gradients_impl

if not hasattr(tf_gradients_impl, '_IsBackpropagatable'):
  from tensorflow.python.ops import gradients_util as tf_gradients_impl


class DeepShap(object):
  """Meant to approximate SHAP values for deep learning models.

  This is an enhanced version of the DeepLIFT algorithm (Deep SHAP) where, similar to Kernel SHAP, we
  approximate the conditional expectations of SHAP values using a selection of background samples.
  Lundberg and Lee, NIPS 2017 showed that the per node attribution rules in DeepLIFT (Shrikumar,
  Greenside, and Kundaje, arXiv 2017) can be chosen to approximate Shapley values. By integrating
  over many backgound samples Deep estimates approximate SHAP values such that they sum
  up to the difference between the expected model output on the passed background samples and the
  current model output (f(x) - E[f(x)]).
  """

  def __init__(self,
               inputs,
               output,
               data,
               session=None,
               learning_phase_flags=None):
    """An explainer object for a deep model using a given background dataset.

    Note that the complexity of the method scales linearly with the number of background data
    samples. Passing the entire training dataset as `data` will give very accurate expected
    values, but be unreasonably expensive. The variance of the expectation estimates scale by
    roughly 1/sqrt(N) for N background data samples. So 100 samples will give a good estimate,
    and 1000 samples a very good estimate of the expected values.

    Parameters
    ----------
    inputs : [tf.Operation]
    output : tf.Operation
        A pair of TensorFlow operations (or a list and an op) that
        specifies the input and output of the model to be explained. Note that SHAP values
        are specific to a single output value, so you get an explanation for each element of
        the output tensor (which must be a flat rank one vector).

    data : [numpy.array] or [pandas.DataFrame] or function
        The background dataset to use for integrating out features. DeepExplainer integrates
        over all these samples for each explanation. The data passed here must match the input
        operations given to the model. If a function is supplied, it must be a function that
        takes a particular input example and generates the background dataset for that example
    session : None or tensorflow.Session
        The TensorFlow session that has the model we are explaining. If None is passed then
        we do our best to find the right session, first looking for a keras session, then
        falling back to the default TensorFlow session.

    learning_phase_flags : None or list of tensors
        If you have your own custom learning phase flags pass them here. When explaining a prediction
        we need to ensure we are not in training mode, since this changes the behavior of ops like
        batch norm or dropout. If None is passed then we look for tensors in the graph that look like
        learning phase flags. Note that we assume all the flags should
        have a value of False during predictions (and hence explanations).
    """
    self.model_inputs = inputs
    self.model_output = output
    assert type(
        self.model_output
    ) != list, 'The model output to be explained must be a single tensor!'
    assert len(self.model_output.shape
               ) < 3, 'The model output must be a vector or a single value!'
    self.multi_output = True
    if len(self.model_output.shape) == 1:
      self.multi_output = False

    # check if we have multiple inputs
    self.multi_input = True
    if type(self.model_inputs) != list or len(self.model_inputs) == 1:
      self.multi_input = False
      if type(self.model_inputs) != list:
        self.model_inputs = [self.model_inputs]
    if type(data) != list and (hasattr(data, '__call__') == False):
      data = [data]
    self.data = data

    self._vinputs = {
    }  # used to track what op inputs depends on the model inputs
    self.orig_grads = {}

    if session is None:
      try:
        session = tf.compat.v1.keras.backend.get_session()
      except:
        session = tf.keras.backend.get_session()
    self.session = tf.get_default_session() if session is None else session
    self.graph = self.session.graph

    # if no learning phase flags were given we go looking for them
    # ...this will catch the one that keras uses
    # we need to find them since we want to make sure learning phase flags are set to False
    if learning_phase_flags is None:
      self.learning_phase_ops = []
      for op in self.graph.get_operations():
        if 'learning_phase' in op.name and op.type == 'Const' and len(
            op.outputs[0].shape) == 0:
          if op.outputs[0].dtype == tf.bool:
            self.learning_phase_ops.append(op)
      self.learning_phase_flags = [
          op.outputs[0] for op in self.learning_phase_ops
      ]
    else:
      self.learning_phase_ops = [t.op for t in learning_phase_flags]

    # save the expected output of the model
    # if self.data is a function, set self.expected_value to None
    if (hasattr(self.data, '__call__')):
      self.expected_value = None
    else:
      if self.data[0].shape[0] > 5000:
        warnings.warn(
            'You have provided over 5k background samples! For better performance consider using smaller random sample.'
        )
      self.expected_value = self.run(self.model_output, self.model_inputs,
                                     self.data).mean(0)

    self._init_between_tensors(self.model_output.op, self.model_inputs)

    # make a blank array that will get lazily filled in with the SHAP value computation
    # graphs for each output. Lazy is important since if there are 1000 outputs and we
    # only explain the top 5 it would be a waste to build graphs for the other 995
    if not self.multi_output:
      self.phi_symbolics = [None]
    else:
      noutputs = self.model_output.shape.as_list()[1]
      if noutputs is not None:
        self.phi_symbolics = [None for i in range(noutputs)]
      else:
        raise Exception(
            'The model output tensor to be explained cannot have a static shape in dim 1 of None!'
        )

  def run(self, out, model_inputs, X):
    """Runs the model while also setting the learning phase flags to False."""
    feed_dict = dict(zip(model_inputs, X))
    for t in self.learning_phase_flags:
      feed_dict[t] = False
    return self.session.run(out, feed_dict)

  def phi_symbolic(self, i):
    """Get the SHAP value computation graph for a given model output."""
    if self.phi_symbolics[i] is None:

      def anon():
        out = self.model_output[:,
                                i] if self.multi_output else self.model_output
        return tf.gradients(out, self.model_inputs)

      self.phi_symbolics[i] = self.execute_with_overridden_gradients(anon)

    return self.phi_symbolics[i]

  def custom_grad(self, op, *grads):
    """Passes a gradient op creation request to the correct handler."""
    type_name = op.type[5:] if op.type.startswith('shap_') else op.type
    out = op_handlers[type_name](
        self, op, *grads)  # we cut off the shap_ prefex before the lookup
    return out

  def execute_with_overridden_gradients(self, f):
    # replace the gradients for all the non-linear activations
    # we do this by hacking our way into the registry (TODO: find a public API for this if it exists)
    reg = tf_ops._gradient_registry._registry
    ops_not_in_registry = ['TensorListReserve']
    # NOTE: location_tag taken from tensorflow source for None type ops
    location_tag = ('UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN')
    # TODO: unclear why some ops are not in the registry with TF 2.0 like TensorListReserve
    for non_reg_ops in ops_not_in_registry:
      reg[non_reg_ops] = {'type': None, 'location': location_tag}
    for n in op_handlers:
      if n in reg:
        self.orig_grads[n] = reg[n]['type']
        reg['shap_' + n] = {
            'type': self.custom_grad,
            'location': reg[n]['location']
        }
        reg[n]['type'] = self.custom_grad

    # In TensorFlow 1.10 they started pruning out nodes that they think can't be backpropped
    # unfortunately that includes the index of embedding layers so we disable that check here
    if hasattr(tf_gradients_impl, '_IsBackpropagatable'):
      orig_IsBackpropagatable = tf_gradients_impl._IsBackpropagatable
      tf_gradients_impl._IsBackpropagatable = lambda tensor: True

    # define the computation graph for the attribution values using a custom gradient-like computation
    try:
      out = f()
    finally:
      # reinstate the backpropagatable check
      if hasattr(tf_gradients_impl, '_IsBackpropagatable'):
        tf_gradients_impl._IsBackpropagatable = orig_IsBackpropagatable

      # restore the original gradient definitions
      for n in op_handlers:
        if n in reg:
          del reg['shap_' + n]
          reg[n]['type'] = self.orig_grads[n]
      for non_reg_ops in ops_not_in_registry:
        del reg[non_reg_ops]
    return out

  def shap_values(self,
                  X,
                  ranked_outputs=None,
                  output_rank_order='max',
                  check_additivity=True):
    """Return approximate SHAP values for the model applied to the data given by X.

    Parameters
    ----------
    X : list, numpy.array, or pandas.DataFrame
        A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to
        explain the model's output.

    ranked_outputs : None or int
        If ranked_outputs is None then we explain all the outputs in a multi-output model. If
        ranked_outputs is a positive integer then we only explain that many of the top model
        outputs (where "top" is determined by output_rank_order). Note that this causes a pair
        of values to be returned (shap_values, indexes), where shap_values is a list of numpy
        arrays for each of the output ranks, and indexes is a matrix that indicates for each sample
        which output indexes were choses as "top".

    output_rank_order : "max", "min", or "max_abs"
        How to order the model outputs when using ranked_outputs, either by maximum, minimum, or
        maximum absolute value.

    Returns
    -------
    array or list
        For a models with a single output this returns a tensor of SHAP values with the same shape
        as X. For a model with multiple outputs this returns a list of SHAP value tensors, each of
        which are the same shape as X. If ranked_outputs is None then this list of tensors matches
        the number of model outputs. If ranked_outputs is a positive integer a pair is returned
        (shap_values, indexes), where shap_values is a list of tensors with a length of
        ranked_outputs, and indexes is a matrix that indicates for each sample which output indexes
        were chosen as "top".
    """
    # check if we have multiple inputs
    if not self.multi_input:
      if type(X) == list and len(X) != 1:
        assert False, 'Expected a single tensor as model input!'
      elif type(X) != list:
        X = [X]
    else:
      assert type(X) == list, 'Expected a list of model inputs!'
    assert len(self.model_inputs) == len(
        X
    ), 'Number of model inputs (%d) does not match the number given (%d)!' % (
        len(self.model_inputs), len(X))

    # rank and determine the model outputs that we will explain
    if ranked_outputs is not None and self.multi_output:
      model_output_values = self.run(self.model_output, self.model_inputs, X)

      if output_rank_order == 'max':
        model_output_ranks = np.argsort(-model_output_values)
      elif output_rank_order == 'min':
        model_output_ranks = np.argsort(model_output_values)
      elif output_rank_order == 'max_abs':
        model_output_ranks = np.argsort(np.abs(model_output_values))
      else:
        assert False, 'output_rank_order must be max, min, or max_abs!'
      model_output_ranks = model_output_ranks[:, :ranked_outputs]
    else:
      model_output_ranks = np.tile(
          np.arange(len(self.phi_symbolics)), (X[0].shape[0], 1))

    # compute the attributions
    output_phis = []
    for i in range(model_output_ranks.shape[1]):
      phis = []
      for k in range(len(X)):
        phis.append(np.zeros(X[k].shape))
      for j in range(X[0].shape[0]):
        if (hasattr(self.data, '__call__')):
          bg_data = self.data([X[l][j] for l in range(len(X))])
          if type(bg_data) != list:
            bg_data = [bg_data]
        else:
          bg_data = self.data

        # tile the inputs to line up with the background data samples
        tiled_X = [
            np.tile(X[l][j:j + 1], (bg_data[l].shape[0],) +
                    tuple([1
                           for k in range(len(X[l].shape) - 1)]))
            for l in range(len(X))
        ]

        # we use the first sample for the current sample and the rest for the references
        joint_input = [
            np.concatenate([tiled_X[l], bg_data[l]], 0) for l in range(len(X))
        ]

        # run attribution computation graph
        feature_ind = model_output_ranks[j, i]
        sample_phis = self.run(
            self.phi_symbolic(feature_ind), self.model_inputs, joint_input)

        # assign the attributions to the right part of the output arrays
        for l in range(len(X)):
          phis[l][j] = (sample_phis[l][bg_data[l].shape[0]:] *
                        (X[l][j] - bg_data[l])).mean(0)

      output_phis.append(phis[0] if not self.multi_input else phis)

    # check that the SHAP values sum up to the model output
    if check_additivity:
      model_output = self.run(self.model_output, self.model_inputs, X)
      for l in range(len(self.expected_value)):
        if not self.multi_input:
          diffs = model_output[:,
                               l] - self.expected_value[l] - output_phis[l].sum(
                                   axis=tuple(range(1, output_phis[l].ndim)))
        else:
          diffs = model_output[:, l] - self.expected_value[l]
          for i in range(len(output_phis[l])):
            diffs -= output_phis[l][i].sum(
                axis=tuple(range(1, output_phis[l][i].ndim)))
        assert np.abs(
          diffs).max() < 1e-2, "The SHAP explanations do not sum up to the model's output! This is either because of a " \
                               'rounding error or because an operator in your computation graph was not fully supported. If ' \
                               'the sum difference of %f is significant compared the scale of your model outputs please post ' \
                               'as a github issue, with a reproducible example if possible so we can debug it.' % np.abs(
          diffs).max()

    if not self.multi_output:
      return output_phis[0]
    elif ranked_outputs is not None:
      return output_phis, model_output_ranks
    else:
      return output_phis

  def _init_between_tensors(self, out_op, model_inputs):
    # find all the operations in the graph between our inputs and outputs
    tensor_blacklist = tensors_blocked_by_false(
        self.learning_phase_ops)  # don't follow learning phase branches
    dependence_breakers = [
        k for k in op_handlers if op_handlers[k] == break_dependence
    ]
    back_ops = backward_walk_ops([out_op], tensor_blacklist,
                                 dependence_breakers)
    start_ops = []
    for minput in model_inputs:
      for op in minput.consumers():
        start_ops.append(op)
    self.between_ops = forward_walk_ops(
        start_ops, tensor_blacklist, dependence_breakers, within_ops=back_ops)

    # note all the tensors that are on the path between the inputs and the output
    self.between_tensors = {}
    for op in self.between_ops:
      for t in op.outputs:
        self.between_tensors[t.name] = True
    for t in model_inputs:
      self.between_tensors[t.name] = True

    # save what types are being used
    self.used_types = {}
    for op in self.between_ops:
      self.used_types[op.type] = True

  def _variable_inputs(self, op):
    """Return which inputs of this operation are variable (i.e. depend on the model inputs)."""
    if op not in self._vinputs:
      out = np.zeros(len(op.inputs), dtype=np.bool)
      for i, t in enumerate(op.inputs):
        out[i] = t.name in self.between_tensors
      self._vinputs[op] = out
    return self._vinputs[op]


def tensors_blocked_by_false(ops):
  """Follows a set of ops assuming their value is False and find blocked Switch paths.

  This is used to prune away parts of the model graph that are only used during the training
  phase (like dropout, batch norm, etc.).
  """
  blocked = []

  def recurse(op):
    if op.type == 'Switch':
      blocked.append(
          op.outputs[1]
      )  # the true path is blocked since we assume the ops we trace are False
    else:
      for out in op.outputs:
        for c in out.consumers():
          recurse(c)

  for op in ops:
    recurse(op)

  return blocked


def backward_walk_ops(start_ops, tensor_blacklist, op_type_blacklist):
  found_ops = []
  op_stack = [op for op in start_ops]
  while len(op_stack) > 0:
    op = op_stack.pop()
    if op.type not in op_type_blacklist and op not in found_ops:
      found_ops.append(op)
      for input in op.inputs:
        if input not in tensor_blacklist:
          op_stack.append(input.op)
  return found_ops


def forward_walk_ops(start_ops, tensor_blacklist, op_type_blacklist,
                     within_ops):
  found_ops = []
  op_stack = [op for op in start_ops]
  while len(op_stack) > 0:
    op = op_stack.pop()
    if op.type not in op_type_blacklist and op in within_ops and op not in found_ops:
      found_ops.append(op)
      for out in op.outputs:
        if out not in tensor_blacklist:
          for c in out.consumers():
            op_stack.append(c)
  return found_ops


def linearity_1d_nonlinearity_2d(input_ind0, input_ind1, op_func):

  def handler(explainer, op, *grads):
    var = explainer._variable_inputs(op)
    if var[input_ind0] and not var[input_ind1]:
      return linearity_1d_handler(input_ind0, explainer, op, *grads)
    elif var[input_ind1] and not var[input_ind0]:
      return linearity_1d_handler(input_ind1, explainer, op, *grads)
    elif var[input_ind0] and var[input_ind1]:
      return nonlinearity_2d_handler(input_ind0, input_ind1, op_func, explainer,
                                     op, *grads)
    else:
      return [None for _ in op.inputs
              ]  # no inputs vary, we must be hidden by a switch function

  return handler


def nonlinearity_1d_nonlinearity_2d(input_ind0, input_ind1, op_func):

  def handler(explainer, op, *grads):
    var = explainer._variable_inputs(op)
    if var[input_ind0] and not var[input_ind1]:
      return nonlinearity_1d_handler(input_ind0, explainer, op, *grads)
    elif var[input_ind1] and not var[input_ind0]:
      return nonlinearity_1d_handler(input_ind1, explainer, op, *grads)
    elif var[input_ind0] and var[input_ind1]:
      return nonlinearity_2d_handler(input_ind0, input_ind1, op_func, explainer,
                                     op, *grads)
    else:
      return [None for _ in op.inputs
              ]  # no inputs vary, we must be hidden by a switch function

  return handler


def nonlinearity_1d(input_ind):

  def handler(explainer, op, *grads):
    return nonlinearity_1d_handler(input_ind, explainer, op, *grads)

  return handler


def nonlinearity_1d_handler(input_ind, explainer, op, *grads):
  # make sure only the given input varies
  op_inputs = op.inputs
  if op_inputs is None:
    op_inputs = op.outputs[0].op.inputs

  for i in range(len(op_inputs)):
    if i != input_ind:
      assert not explainer._variable_inputs(
          op)[i], str(i) + 'th input to ' + op.name + ' cannot vary!'

  xin0, rin0 = tf.split(op_inputs[input_ind], 2)
  xout, rout = tf.split(op.outputs[input_ind], 2)
  delta_in0 = xin0 - rin0
  if delta_in0.shape is None:
    dup0 = [2, 1]
  else:
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]
  out = [None for _ in op_inputs]
  if op.type.startswith('shap_'):
    op.type = op.type[5:]
  orig_grad = explainer.orig_grads[op.type](op, grads[0])
  out[input_ind] = tf.where(
      tf.tile(tf.abs(delta_in0), dup0) < 1e-6,
      orig_grad[input_ind] if len(op_inputs) > 1 else orig_grad,
      grads[0] * tf.tile((xout - rout) / delta_in0, dup0))
  return out


def nonlinearity_2d_handler(input_ind0, input_ind1, op_func, explainer, op,
                            *grads):
  assert input_ind0 == 0 and input_ind1 == 1, "TODO: Can't yet handle double inputs that are not first!"
  xout, rout = tf.split(op.outputs[0], 2)
  in0 = op.inputs[input_ind0]
  in1 = op.inputs[input_ind1]
  xin0, rin0 = tf.split(in0, 2)
  xin1, rin1 = tf.split(in1, 2)
  delta_in0 = xin0 - rin0
  delta_in1 = xin1 - rin1
  dup0 = [2] + [1 for i in delta_in0.shape[1:]]
  out10 = op_func(xin0, rin1)
  out01 = op_func(rin0, xin1)
  out11, out00 = xout, rout
  out0 = 0.5 * (out11 - out01 + out10 - out00)
  out0 = grads[0] * tf.tile(out0 / delta_in0, dup0)
  out1 = 0.5 * (out11 - out10 + out01 - out00)
  out1 = grads[0] * tf.tile(out1 / delta_in1, dup0)

  # Avoid divide by zero nans
  out0 = tf.where(
      tf.abs(tf.tile(delta_in0, dup0)) < 1e-7, tf.zeros_like(out0), out0)
  out1 = tf.where(
      tf.abs(tf.tile(delta_in1, dup0)) < 1e-7, tf.zeros_like(out1), out1)

  # see if due to broadcasting our gradient shapes don't match our input shapes
  if (np.any(np.array(out1.shape) != np.array(in1.shape))):
    broadcast_index = np.where(
        np.array(out1.shape) != np.array(in1.shape))[0][0]
    out1 = tf.reduce_sum(out1, axis=broadcast_index, keepdims=True)
  elif (np.any(np.array(out0.shape) != np.array(in0.shape))):
    broadcast_index = np.where(
        np.array(out0.shape) != np.array(in0.shape))[0][0]
    out0 = tf.reduce_sum(out0, axis=broadcast_index, keepdims=True)

  return [out0, out1]


def softmax(explainer, op, *grads):
  """Just decompose softmax into its components and recurse, we can handle all of them :)

  We assume the 'axis' is the last dimension because the TF codebase swaps the 'axis' to
  the last dimension before the softmax op if 'axis' is not already the last dimension.
  We also don't subtract the max before tf.exp for numerical stability since that might
  mess up the attributions and it seems like TensorFlow doesn't define softmax that way
  (according to the docs)
  """
  in0 = op.inputs[0]
  in0_max = tf.reduce_max(in0, axis=-1, keepdims=True, name='in0_max')
  in0_centered = in0 - in0_max
  evals = tf.exp(in0_centered, name='custom_exp')
  rsum = tf.reduce_sum(evals, axis=-1, keepdims=True)
  div = evals / rsum

  # mark these as in-between the inputs and outputs
  for op in [evals.op, rsum.op, div.op, in0_centered.op]:
    for t in op.outputs:
      if t.name not in explainer.between_tensors:
        explainer.between_tensors[t.name] = False

  out = tf.gradients(div, in0_centered, grad_ys=grads[0])[0]

  # remove the names we just added
  for op in [evals.op, rsum.op, div.op, in0_centered.op]:
    for t in op.outputs:
      if explainer.between_tensors[t.name] is False:
        del explainer.between_tensors[t.name]

  # rescale to account for our shift by in0_max (which we did for numerical stability)
  xin0, rin0 = tf.split(in0, 2)
  xin0_centered, rin0_centered = tf.split(in0_centered, 2)
  delta_in0 = xin0 - rin0
  dup0 = [2] + [1 for i in delta_in0.shape[1:]]
  return tf.where(
      tf.tile(tf.abs(delta_in0), dup0) < 1e-6, out,
      out * tf.tile((xin0_centered - rin0_centered) / delta_in0, dup0))


def maxpool(explainer, op, *grads):
  xin0, rin0 = tf.split(op.inputs[0], 2)
  xout, rout = tf.split(op.outputs[0], 2)
  delta_in0 = xin0 - rin0
  dup0 = [2] + [1 for i in delta_in0.shape[1:]]
  cross_max = tf.maximum(xout, rout)
  diffs = tf.concat([cross_max - rout, xout - cross_max], 0)
  if op.type.startswith('shap_'):
    op.type = op.type[5:]
  xmax_pos, rmax_pos = tf.split(
      explainer.orig_grads[op.type](op, grads[0] * diffs), 2)
  return tf.tile(
      tf.where(
          tf.abs(delta_in0) < 1e-7, tf.zeros_like(delta_in0),
          (xmax_pos + rmax_pos) / delta_in0), dup0)


def gather(explainer, op, *grads):
  # params = op.inputs[0]
  indices = op.inputs[1]
  # axis = op.inputs[2]
  var = explainer._variable_inputs(op)
  if var[1] and not var[0]:
    assert len(indices.shape
               ) == 2, 'Only scalar indices supported right now in GatherV2!'

    xin1, rin1 = tf.split(tf.cast(op.inputs[1], tf.float32), 2)
    xout, rout = tf.split(op.outputs[0], 2)
    dup_in1 = [2] + [1 for i in xin1.shape[1:]]
    dup_out = [2] + [1 for i in xout.shape[1:]]
    delta_in1_t = tf.tile(xin1 - rin1, dup_in1)
    out_sum = tf.reduce_sum(
        grads[0] * tf.tile(xout - rout, dup_out),
        list(range(len(indices.shape), len(grads[0].shape))))
    if op.type == 'ResourceGather':
      return [
          None,
          tf.where(
              tf.abs(delta_in1_t) < 1e-6, tf.zeros_like(delta_in1_t),
              out_sum / delta_in1_t)
      ]
    return [
        None,
        tf.where(
            tf.abs(delta_in1_t) < 1e-6, tf.zeros_like(delta_in1_t),
            out_sum / delta_in1_t), None
    ]
  elif var[0] and not var[1]:
    if op.type.startswith('shap_'):
      op.type = op.type[5:]
    return [explainer.orig_grads[op.type](op, grads[0]),
            None]  # linear in this case
  else:
    assert False, 'Axis not yet supported to be varying for gather op!'


def linearity_1d(input_ind):

  def handler(explainer, op, *grads):
    return linearity_1d_handler(input_ind, explainer, op, *grads)

  return handler


def linearity_1d_handler(input_ind, explainer, op, *grads):
  # make sure only the given input varies (negative means only that input cannot vary, and is measured from the end of the list)
  for i in range(len(op.inputs)):
    if i != input_ind:
      assert not explainer._variable_inputs(
          op)[i], str(i) + 'th input to ' + op.name + ' cannot vary!'
  if op.type.startswith('shap_'):
    op.type = op.type[5:]
  return explainer.orig_grads[op.type](op, *grads)


def linearity_with_excluded(input_inds):

  def handler(explainer, op, *grads):
    return linearity_with_excluded_handler(input_inds, explainer, op, *grads)

  return handler


def linearity_with_excluded_handler(input_inds, explainer, op, *grads):
  # make sure the given inputs don't vary (negative is measured from the end of the list)
  for i in range(len(op.inputs)):
    if i in input_inds or i - len(op.inputs) in input_inds:
      assert not explainer._variable_inputs(
          op)[i], str(i) + 'th input to ' + op.name + ' cannot vary!'
  if op.type.startswith('shap_'):
    op.type = op.type[5:]
  return explainer.orig_grads[op.type](op, *grads)


def passthrough(explainer, op, *grads):
  if op.type.startswith('shap_'):
    op.type = op.type[5:]
  return explainer.orig_grads[op.type](op, *grads)


def break_dependence(explainer, op, *grads):
  """This function name is used to break attribution dependence in the graph traversal.

  These operation types may be connected above input data values in the graph but their outputs
  don't depend on the input values (for example they just depend on the shape).
  """
  return [None for _ in op.inputs]


op_handlers = {}

# ops that are always linear
op_handlers['Identity'] = passthrough
op_handlers['StridedSlice'] = passthrough
op_handlers['Squeeze'] = passthrough
op_handlers['ExpandDims'] = passthrough
op_handlers['Pack'] = passthrough
op_handlers['BiasAdd'] = passthrough
op_handlers['Unpack'] = passthrough
op_handlers['Add'] = passthrough
op_handlers['Sub'] = passthrough
op_handlers['Merge'] = passthrough
op_handlers['Sum'] = passthrough
op_handlers['Mean'] = passthrough
op_handlers['Cast'] = passthrough
op_handlers['Transpose'] = passthrough
op_handlers['Enter'] = passthrough
op_handlers['Exit'] = passthrough
op_handlers['NextIteration'] = passthrough
op_handlers['Tile'] = passthrough
op_handlers['TensorArrayScatterV3'] = passthrough
op_handlers['TensorArrayReadV3'] = passthrough
op_handlers['TensorArrayWriteV3'] = passthrough

# ops that don't pass any attributions to their inputs
op_handlers['Shape'] = break_dependence
op_handlers['RandomUniform'] = break_dependence
op_handlers['ZerosLike'] = break_dependence
# op_handlers["StopGradient"] = break_dependence # this allows us to stop attributions when we want to (like softmax re-centering)

# ops that are linear and only allow a single input to vary
op_handlers['Reshape'] = linearity_1d(0)
op_handlers['Pad'] = linearity_1d(0)
op_handlers['ReverseV2'] = linearity_1d(0)
op_handlers['ConcatV2'] = linearity_with_excluded([-1])
op_handlers['Conv2D'] = linearity_1d(0)
op_handlers['Switch'] = linearity_1d(0)
op_handlers['AvgPool'] = linearity_1d(0)
op_handlers['FusedBatchNorm'] = linearity_1d(0)

# ops that are nonlinear and only allow a single input to vary
op_handlers['Relu'] = nonlinearity_1d(0)
op_handlers['Elu'] = nonlinearity_1d(0)
op_handlers['Sigmoid'] = nonlinearity_1d(0)
op_handlers['Tanh'] = nonlinearity_1d(0)
op_handlers['Softplus'] = nonlinearity_1d(0)
op_handlers['Exp'] = nonlinearity_1d(0)
op_handlers['ClipByValue'] = nonlinearity_1d(0)
op_handlers['Rsqrt'] = nonlinearity_1d(0)
op_handlers['Square'] = nonlinearity_1d(0)
op_handlers['Max'] = nonlinearity_1d(0)

# ops that are nonlinear and allow two inputs to vary
op_handlers['SquaredDifference'] = nonlinearity_1d_nonlinearity_2d(
    0, 1, lambda x, y: (x - y) * (x - y))
op_handlers['Minimum'] = nonlinearity_1d_nonlinearity_2d(
    0, 1, lambda x, y: tf.minimum(x, y))
op_handlers['Maximum'] = nonlinearity_1d_nonlinearity_2d(
    0, 1, lambda x, y: tf.maximum(x, y))

# ops that allow up to two inputs to vary are are linear when only one input varies
op_handlers['Mul'] = linearity_1d_nonlinearity_2d(0, 1, lambda x, y: x * y)
op_handlers['RealDiv'] = linearity_1d_nonlinearity_2d(0, 1, lambda x, y: x / y)
op_handlers['MatMul'] = linearity_1d_nonlinearity_2d(
    0, 1, lambda x, y: tf.matmul(x, y))

# ops that need their own custom attribution functions
op_handlers['GatherV2'] = gather
op_handlers['ResourceGather'] = gather
op_handlers['MaxPool'] = maxpool
op_handlers['Softmax'] = softmax