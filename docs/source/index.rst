.. easy_rec documentation master file, created by
   sphinx-quickstart on Wed Nov 11 14:37:57 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to easy_rec's documentation!
====================================

.. toctree::
   :maxdepth: 1
   :caption: USER GUIDE

   intro
   quick_start

.. toctree::
   :maxdepth: 2
   :caption: DATA & FEATURE

   feature/data
   feature/feature
   feature/excel_config
   feature/rtp_fg

.. toctree::
   :maxdepth: 3
   :caption: MODEL

   models/recall
   models/rank
   models/multi_target
   models/user_define

.. toctree::
   :maxdepth: 2
   :caption: TRAIN & EVAL & EXPORT

   train
   incremental_train
   eval
   export
   kd
   optimize

.. toctree::
   :maxdepth: 2
   :caption: PREDICT

   predict/MaxCompute 离线预测
   predict/Local 离线预测
   predict/在线预测

.. toctree::
   :maxdepth: 2
   :caption: AUTOML

   automl/hpo_pai
   automl/hpo_emr
   automl/auto_cross_emr

.. toctree::
   :maxdepth: 10
   :caption: API

   api/easy_rec.python.main
   api/easy_rec.python.model
   api/easy_rec.python.input
   api/easy_rec.python.layers
   api/easy_rec.python.core
   api/easy_rec.python.feature_column
   api/easy_rec.python.inference
   api/easy_rec.python.builders
   api/easy_rec.python.utils

.. toctree::
   :maxdepth: 1
   :caption: DEVELOP

   develop
   release

.. toctree::
   :maxdepth: 2
   :caption: REFERENCE

   reference
   faq
   tf_on_yarn
   get_role_arn
   benchmark


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
