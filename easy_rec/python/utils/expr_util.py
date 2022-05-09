from collections import deque


def _process_multi_expr(expr):
  expr = expr.strip()
  size = len(expr)
  idx = 0
  two_expr = ['>=', '<=', '==']
  expr_list = []
  while (idx < size):
    if idx + 2 <= size and expr[idx:idx + 2] in two_expr:
      expr_list.append(expr[idx:idx + 2])
      idx += 2
    else:
      expr_list.append(expr[idx])
      idx += 1
  return expr_list


def _process_enum(enum, input_names, prefix=''):
  enum = enum.strip()
  if enum in input_names:
    enum = "parsed_dict['%s']" % (prefix + enum)
  return enum


def _get_expression_list(expression, input_names, prefix=''):
  ops = [
      '+', '-', '*', '/', '(', ')', '>', '>=', '<', '<=', '==', '=', '&', '|'
  ]
  expression_list = []
  eunm = ''
  pre_expr = ''

  for i in expression:
    if i in ops:
      if eunm:
        expression_list.append(_process_enum(eunm, input_names, prefix=prefix))
        eunm = ''
      pre_expr += i
    else:
      eunm += i
      if pre_expr:
        expression_list.extend(_process_multi_expr(pre_expr))
        pre_expr = ''
  if eunm:
    expression_list.append(_process_enum(eunm, input_names, prefix=prefix))
  if pre_expr:
    expression_list.extend(_process_multi_expr(pre_expr))

  final_expression_list = ['']
  ops = ['(', ')', '>=', '<=', '==', '>', '<', '&', '|']
  for expr in expression_list:
    if expr in ops:
      final_expression_list.append(expr)
    elif final_expression_list[-1] not in ops:
      final_expression_list[-1] += expr
    else:
      final_expression_list.append(expr)
  final_expression_list = [expr for expr in final_expression_list if expr]
  return final_expression_list


def _solve(enum, sign, stack):
  if len(stack) == 0 or enum == '' or sign == '':
    return enum
  op1 = stack.pop()
  op2 = enum
  if sign == '>':
    result = 'tf.greater(%s, %s)' % (op1, op2)
  elif sign == '>=':
    result = 'tf.greater_equal(%s, %s)' % (op1, op2)
  elif sign == '<':
    result = 'tf.less(%s, %s)' % (op1, op2)
  elif sign == '<=':
    result = 'tf.less_equal(%s, %s)' % (op1, op2)
  elif sign == '==':
    result = 'tf.equal(%s, %s)' % (op1, op2)
  elif sign == '&':
    result = '%s & %s' % (op1, op2)
  elif sign == '|':
    result = '%s | %s' % (op1, op2)
  else:
    assert False
  return result


def _expression_eval(expr_list):
  ops = ['>', '>=', '<', '<=', '==', '&', '|', '(', ')']
  stack = deque()
  sign = ''
  operand = ''
  for c in expr_list:
    if c == ' ':
      continue
    elif c not in ops:
      operand = c
    elif c == '(':
      stack.append(sign)
      sign = ''
    else:
      result = _solve(operand, sign, stack)
      operand = ''
      if c == ')':
        sign = stack.pop()
        operand = _solve(result, sign, stack)
        sign = ''
      else:
        sign = c
        stack.append(result)
  expr_str = _solve(operand, sign, stack)
  return expr_str


def get_expression(expression, input_names, prefix=''):
  expression_list = _get_expression_list(expression, input_names, prefix=prefix)
  expression = _expression_eval(expression_list)
  return expression
