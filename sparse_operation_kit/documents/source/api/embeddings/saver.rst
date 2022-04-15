SparseOperationKit Embedding Saver
==================================
Currently, you must explicitly call this saver to dump or load
trainable parameters for the trainable variables in those 
embedding layers. And other trainable variables are still managed
by the DL framework.

.. autoclass:: sparse_operation_kit.saver.Saver.Saver
   :members: 