# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Load_class.py tools for loading classes."""

import inspect
import logging
import os
import pkgutil
import pydoc
import traceback
from abc import ABCMeta

import six
import tensorflow as tf

import easy_rec
from easy_rec.python.utils import compat


def python_file_to_module(python_file):
  mod = python_file.strip('/').replace('/', '.')
  if mod.endswith('.py'):
    mod = mod[:-3]
  return mod


def load_by_path(path):
  """Load functions or modules or classes.

  Args:
    path: path to modules or functions or classes,
        such as: tf.nn.relu

  Return:
    modules or functions or classes
  """
  path = path.strip()
  if path == '' or path is None:
    return None
  if 'lambda' in path:
    return eval(path)
  components = path.split('.')
  if components[0] == 'tf':
    components[0] = 'tensorflow'
  path = '.'.join(components)
  try:
    return pydoc.locate(path)
  except pydoc.ErrorDuringImport:
    logging.error('load %s failed: %s' % (path, traceback.format_exc()))
    return None


def _get_methods(aClass):

  def should_track(func_name):
    return func_name == '__init__' or func_name[0] != '_'

  names = sorted(dir(aClass), key=str.lower)
  attrs = [(n, getattr(aClass, n)) for n in names if should_track(n)]
  # in python3 , unbound class method is function while is
  # method in python2
  if compat.in_python3():
    return dict((n, a) for n, a in attrs if inspect.isfunction(a))
  else:
    return dict((n, a) for n, a in attrs if inspect.ismethod(a))


def _get_method_declare(aMethod):
  try:
    name = aMethod.__name__
    if compat.in_python3():
      sig_str = str(inspect.signature(aMethod))
      return sig_str
    else:
      spec = inspect.getargspec(aMethod)
      args = inspect.formatargspec(spec.args, spec.varargs, spec.keywords,
                                   spec.defaults)
      return '%s%s' % (name, args)
  except TypeError:
    return '%s(cls, ...)' % name


def check_class(cls, impl_cls, function_names=None):
  """Check implemented class is valid according to template class.

  if function signature is not the same, exception will be raised.

  Args:
    cls: class which declares functions that need users to implement
    impl_cls: user implemented class
    function_names: if not None, will only check these funtions and their signature
  """
  missing = {}

  ours = _get_methods(cls)
  theirs = _get_methods(impl_cls)

  for name, method in six.iteritems(ours):
    if function_names is not None and name not in function_names:
      continue
    if name not in theirs:
      missing[name + '()'] = 'not implemented'
      continue
    ourf = _get_method_declare(method)
    theirf = _get_method_declare(theirs[name])

    if not (ourf == theirf):
      missing[name + '()'] = 'method signature differs'

  if len(missing) > 0:
    raise Exception('incompatible Implementation-implementation %s: %s' %
                    (impl_cls.__class__.__name__, missing))


def import_pkg(pkg_info, prefix_to_remove=None):
  """Import package.

  Args:
    pkg_info: pkgutil.ModuleInfo object
    prefix_to_remove: the package prefix to be removed
  """
  package_path = pkg_info[0].path
  if prefix_to_remove is not None:
    package_path = package_path.replace(prefix_to_remove, '')
  mod_name = pkg_info[1]

  if package_path.startswith('/'):
    # absolute path file, we should use relative import
    mod = pkg_info[0].find_module(mod_name)
    if mod is not None:
      # skip those test files in easyrec
      if not mod_name.endswith('_test'):
        mod.load_module(pkg_info[1])
    else:
      raise Exception('import module %s failed' % (package_path + mod_name))
  else:
    # use similar import methods as the import keyword
    module_path = os.path.join(package_path, mod_name).replace('/', '.')
    # skip those test files
    if not mod_name.endswith('_test'):
      try:
        __import__(module_path)
      except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        raise ValueError('import module %s failed: %s' % (module_path, str(e)))


def auto_import(user_path=None):
  """Auto import python files so that register_xxx decorator will take effect.

  By default, we will import files in pre-defined directory and import all
  files recursively in user_dir

  Args:
    user_path: directory or file that store user-defined python code, by default we wiil only
      search file in current directory
  """
  # True False indicates import recursively or not
  pre_defined_dirs = [
      ('easy_rec/python/model', False),
      ('easy_rec/python/input', False),
  ]

  parent_dir = easy_rec.parent_dir
  prefix_to_remove = None
  # dealing with easy-rec in sited-packages, remove parent directory prefix
  # to make class name starts with easy_rec
  if parent_dir != '':
    for idx in range(len(pre_defined_dirs)):
      pre_defined_dirs[idx] = (os.path.join(parent_dir,
                                            pre_defined_dirs[idx][0]),
                               pre_defined_dirs[idx][1])
    prefix_to_remove = parent_dir + '/'

  if user_path is not None:
    if tf.gfile.IsDirectory(user_path):
      user_dir = user_path
    else:
      user_dir, _ = os.path.split(user_path)
    pre_defined_dirs.append((user_dir, True))

  for dir_path, recursive_import in pre_defined_dirs:
    for pkg_info in pkgutil.iter_modules([dir_path]):
      import_pkg(pkg_info, prefix_to_remove)

    if recursive_import:
      for root, dirs, files in os.walk(dir_path):
        for subdir in dirs:
          dirname = os.path.join(root, subdir)
          for pkg_info in pkgutil.iter_modules([dirname]):
            import_pkg(pkg_info, prefix_to_remove)


def register_class(class_map, class_name, cls):
  assert class_name not in class_map or class_map[class_name] == cls, \
      'confilict class %s , %s is already register to be %s' % (
          cls, class_name, str(class_map[class_name]))
  logging.debug('register class %s' % class_name)
  class_map[class_name] = cls


def get_register_class_meta(class_map, have_abstract_class=True):

  class RegisterABCMeta(ABCMeta):

    def __new__(mcs, name, bases, attrs):
      newclass = super(RegisterABCMeta, mcs).__new__(mcs, name, bases, attrs)
      register_class(class_map, name, newclass)

      @classmethod
      def create_class(cls, name):
        if name in class_map:
          return class_map[name]
        else:
          raise Exception('Class %s is not registered. Available ones are %s' %
                          (name, list(class_map.keys())))

      setattr(newclass, 'create_class', create_class)
      return newclass

  return RegisterABCMeta


def load_keras_layer(name):
  """Load keras layer class.

  Args:
    name: keras layer name

  Return:
    (layer_class, is_customize)
  """
  name = name.strip()
  if name == '' or name is None:
    return None

  path = 'easy_rec.python.layers.keras.' + name
  try:
    cls = pydoc.locate(path)
    if cls is not None:
      return cls, True
    path = 'tensorflow.keras.layers.' + name
    return pydoc.locate(path), False
  except pydoc.ErrorDuringImport:
    print('load keras layer %s failed' % name)
    logging.error('load keras layer %s failed: %s' %
                  (name, traceback.format_exc()))
    return None, False
