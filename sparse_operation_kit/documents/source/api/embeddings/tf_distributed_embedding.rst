TF Distributed Embedding
========================
Wrapper classes to build model-parallel embedding layer
totally with TensorFlow's API. It utilizes `tf.distribute.Strategy`
to do the communication among different GPUs.

.. autoclass:: sparse_operation_kit.embeddings.tf_distributed_embedding.TFDistributedEmbedding
    :members: call