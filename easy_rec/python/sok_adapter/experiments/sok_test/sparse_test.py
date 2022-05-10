import pdb
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(
#    os.path.dirname(os.path.abspath(__file__)), r"../../../")))
import sparse_operation_kit as sok
import tensorflow as tf
from tensorflow.distribute import get_replica_context
import numpy as np

from easy_rec.python.sok_adapter import modify_apply_gradients


global_batch_size = 512
global_embedding_table_size = 1024
emb_vec_size = 8
slot_num = 10
gpu_num = 2
workers = 1
max_nnz = 8

indices_val = np.random.randint(0, global_embedding_table_size, global_batch_size).reshape([-1, 1, 1])
indices = tf.convert_to_tensor(indices_val, dtype=tf.int64)

labels_val = np.random.rand(global_batch_size, slot_num, emb_vec_size)
labels = tf.convert_to_tensor(labels_val, dtype=tf.float32)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    sok_init_op = sok.Init(global_batch_size=global_batch_size)

    model = sok.DistributedEmbedding(
        combiner='mean',
        max_vocabulary_size_per_gpu=(global_embedding_table_size // gpu_num),
        embedding_vec_size=emb_vec_size,
        slot_num=slot_num,
        max_nnz=max_nnz,
        use_hashtable=False,
        key_dtype=indices.dtype,
    )

    emb_opt = tf.keras.optimizers.SGD(learning_rate=0.9)
    #emb_opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    emb_opt = modify_apply_gradients(emb_opt)

def train_step(indices, labels):
    replica_ctx = get_replica_context()
    replica_id = replica_ctx.replica_id_in_sync_group
    local_batch = global_batch_size // gpu_num
    local_indices = indices[replica_id * local_batch: (replica_id + 1) * local_batch]
    local_indices = tf.sparse.from_dense(tf.reshape(local_indices, (-1, 1)))
    local_labels = labels[replica_id * local_batch: (replica_id + 1) * local_batch]
    embedding_vector = model(local_indices)
    dense_layer = tf.layers.Dense(emb_vec_size)
    embedding_vector = dense_layer(embedding_vector)

    loss = tf.reduce_mean((embedding_vector - local_labels) ** 2.0, keepdims=True)
    emb_vars, other_vars = sok.optimizers.utils.split_embedding_variable_from_others(model.trainable_variables)

    replica_emb_vars = []
    replica_emb_grads = []
    for emb_var in emb_vars:
        grads = tf.gradients(loss, [v for v in emb_var.values], colocate_gradients_with_ops=False, unconnected_gradients=tf.UnconnectedGradients.NONE)
        valid_grad = []
        valid_grad_idx = []
        for idx, grad in enumerate(grads):
            if grad is not None:
                valid_grad.append(grad)
                valid_grad_idx.append(idx)

        assert(len(valid_grad) == 1)
        assert(len(valid_grad_idx) == 1)
        valid_grad = valid_grad[0]
        valid_grad_idx = valid_grad_idx[0]
        replica_emb_vars.append(emb_var.values[valid_grad_idx])
        replica_emb_grads.append(valid_grad)

    #grads = tf.gradients(loss, emb_vars,
    #                     colocate_gradients_with_ops=True,
    #                     unconnected_gradients=tf.UnconnectedGradients.NONE)
    # tf.gradients(loss, emb_vars[0].values[0], colocate_gradients_with_ops=True,unconnected_gradients=tf.UnconnectedGradients.NONE)
    loss = tf.nn.compute_average_loss(
        loss, global_batch_size=global_batch_size
    )

    with sok.OptimizerScope(emb_vars):
        sok_emb_update_ops = emb_opt.apply_gradients(zip(replica_emb_grads, replica_emb_vars))
    with tf.control_dependencies([sok_emb_update_ops]):
        loss = tf.identity(loss)
    return loss, replica_emb_grads, emb_vars

replica_loss, replica_emb_grads, emb_vars = strategy.experimental_run_v2(train_step, args=(indices, labels))


# print(replica_loss.values)
# print(replica_emb_grads)

target = replica_loss.values
for g in replica_emb_grads:
   target += g.values

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(sok_init_op)
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        ret = sess.run(target[0])
        print(ret)
