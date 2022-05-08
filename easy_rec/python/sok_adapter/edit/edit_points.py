          /usr/local/lib/python3.8/dist-packages/tensorflow_estimator/python/estimator/estimator.py:1297

          
          import sparse_operation_kit as sok
          from easy_rec.python.sok_adapter import modify_apply_gradients
          sok_init_op = sok.Init(global_batch_size=64)
          sok_instance = sok.DistributedEmbedding(
            combiner='mean',
            max_vocabulary_size_per_gpu=2500000,
            embedding_vec_size=16,
            slot_num=12,
            max_nnz=1000,
            use_hashtable=False,
            key_dtype=dtypes.int64)
          import tensorflow
          emb_opt = tensorflow.keras.optimizers.SGD(learning_rate=0.9)
          emb_opt = modify_apply_gradients(emb_opt)
          ops.add_to_collection('SOK', sok_init_op)
          ops.add_to_collection('SOK', sok_instance)
          ops.add_to_collection('SOK', emb_opt)

          /usr/local/lib/python3.8/dist-packages/tensorflow_estimator/python/estimator/estimator.py:1846

          /usr/local/lib/python3.8/dist-packages/tensorflow_core/python/training/saver.py:868
          self._var_list = [v for v in self._var_list if not "EmbeddingVariable" in v.name]

          /usr/local/lib/python3.8/dist-packages/tensorflow_core/python/training/session_manager.py:295
          sok_init_op = ops.get_collection("SOK")[0]
          import pdb;pdb.set_trace()
          sess.run(sok_init_op)
