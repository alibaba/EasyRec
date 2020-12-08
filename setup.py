# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from setuptools import find_packages
from setuptools import setup


def readme():
  with open('README.md') as f:
    content = f.read()
  return content


version_file = 'easy_rec/version.py'


def get_version():
  with open(version_file, 'r') as f:
    exec(compile(f.read(), version_file, 'exec'))
  return locals()['__version__']


def parse_requirements(fname='requirements.txt'):
  """Parse the package dependencies listed in a requirements file."""

  def parse_line(line):
    """Parse information from a line in a requirements text file."""
    if line.startswith('-r '):
      # Allow specifying requirements in other files
      target = line.split(' ')[1]
      for line in parse_require_file(target):
        yield line
    else:
      yield line

  def parse_require_file(fpath):
    with open(fpath, 'r') as f:
      for line in f.readlines():
        line = line.strip()
        if line and not line.startswith('#'):
          for ll in parse_line(line):
            yield ll

  packages = list(parse_require_file(fname))
  return packages


setup(
    name='easy-rec',
    version=get_version(),
    description='An framework for deep learning on recommendation',
    doc=readme(),
    author='EasyRec Team',
    author_email='easy_rec@alibaba-inc.com',
    url='http://gitlab.alibaba-inc.com/pai_biz_arch/EasyRec',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3'
    ],
    tests_require=parse_requirements('requirements/tests.txt'),
    install_requires=parse_requirements('requirements/runtime.txt'),
    extras_require={
        'all': parse_requirements('requirements.txt'),
        'tests': parse_requirements('requirements/tests.txt'),
    })
