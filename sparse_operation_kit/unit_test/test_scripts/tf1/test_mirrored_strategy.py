import sys, os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), r"../../../")))
import sparse_operation_kit as sok
import tensorflow as tf
from tensorflow.distribute import get_replica_context
import numpy as np


global_batch_size = 512
global_embedding_table_size = 1024
emb_vec_size = 8
slot_num = 1
gpu_num = 8
workers = 1
max_nnz = 8

indices_val = np.random.randint(0, global_embedding_table_size, global_batch_size).reshape([-1, 1, 1])
indices = tf.convert_to_tensor(indices_val, dtype=tf.int64)
labels_val = np.random.rand(global_batch_size, slot_num, max_nnz, emb_vec_size)
labels = tf.convert_to_tensor(labels_val, dtype=tf.float32)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    sok_init_op = sok.Init(global_batch_size=global_batch_size)
    model = sok.All2AllDenseEmbedding(
        max_vocabulary_size_per_gpu=(global_embedding_table_size // gpu_num),
        embedding_vec_size=emb_vec_size,
        slot_num=1,
        nnz_per_slot=1,
        dynamic_input=False,
        use_hashtable=False
    )

def train_step(indices, labels):
    replica_ctx = get_replica_context()
    replica_id = replica_ctx.replica_id_in_sync_group
    local_batch = global_batch_size // gpu_num
    local_indices = indices[replica_id * local_batch : (replica_id + 1) * local_batch]
    local_labels = labels[replica_id * local_batch : (replica_id + 1) * local_batch]
    embedding_vector = model(local_indices)
    loss = tf.reduce_mean((embedding_vector - local_labels) ** 2.0)
    emb_vars, other_vars = sok.optimizers.utils.split_embedding_variable_from_others(model.trainable_variables)
    grads = tf.gradients(loss, emb_vars + other_vars, 
                         colocate_gradients_with_ops=True,
                         unconnected_gradients=tf.UnconnectedGradients.NONE)
    emb_grads = grads[:len(emb_vars)]
    other_grads = grads[len(emb_vars):]
    return loss, emb_grads

replica_loss, replica_emb_grads = strategy.experimental_run_v2(train_step, args=(indices, labels))
print(replica_loss.values)
print(replica_emb_grads)

target = replica_loss.values
for g in replica_emb_grads:
    target += g.values

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(sok_init_op)
    sess.run(tf.global_variables_initializer())

    ret = sess.run(target)
    print(ret)

