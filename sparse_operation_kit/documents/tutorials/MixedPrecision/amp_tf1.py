"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import tensorflow as tf
import sys
sys.path.append("../")
import utility
sys.path.append("../DenseDemo")
import sparse_operation_kit as sok
import horovod.tensorflow as hvd

def main():
    global_batch_size = 1024
    slot_num = 10
    nnz_per_slot = 5

    from tensorflow.python.keras.engine import base_layer_utils
    base_layer_utils.enable_v2_dtype_behavior()

    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)

    dataset = utility.get_dataset(global_batch_size//hvd.size(), read_batchsize=global_batch_size//hvd.size())

    sok_init_op = sok.Init(global_batch_size=global_batch_size)

    model = utility.SOKDenseDemo(max_vocabulary_size_per_gpu=1024,
                                embedding_vec_size=8,
                                slot_num=slot_num,
                                nnz_per_slot=nnz_per_slot,
                                num_dense_layers=0)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    optimizer = sok.tf.keras.mixed_precision.LossScaleOptimizer(optimizer, 1024)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        dtype = loss.dtype
        loss = tf.cast(loss, tf.float32)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)
        return tf.cast(loss, dtype)

    def train_step(inputs, labels):
        logit = model(inputs, training=True)
        loss = _replica_loss(labels, logit)
        scaled_loss = optimizer.get_scaled_loss(loss)
        scaled_gradients = tf.gradients(scaled_loss, model.trainable_variables)
        emb_vars, other_vars =\
            sok.split_embedding_variable_from_others(model.trainable_variables)
        scaled_emb_grads, scaled_other_grads =\
            scaled_gradients[:len(emb_vars)], scaled_gradients[len(emb_vars):]
        emb_grads = optimizer.get_unscaled_gradients(scaled_emb_grads)
        other_grads = optimizer.get_unscaled_gradients(scaled_other_grads)
        other_grads = [hvd.allreduce(grad) for grad in other_grads]
        with sok.OptimizerScope(emb_vars):
            emb_train_op = optimizer.apply_gradients(zip(emb_grads, emb_vars))
        other_train_op = optimizer.apply_gradients(zip(other_grads, other_vars))
        total_loss = hvd.allreduce(loss)
        with tf.control_dependencies([emb_train_op, other_train_op]):
            return tf.identity(total_loss)

    train_iterator = dataset.make_initializable_iterator()
    iterator_init = train_iterator.initializer
    inputs, labels = train_iterator.get_next()

    loss = train_step(inputs, labels)

    init_op = tf.group(tf.global_variables_initializer(), 
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(sok_init_op)
        sess.run([init_op, iterator_init])
        
        for step in range(10):
            loss_v = sess.run(loss)
            if hvd.local_rank() == 0:
                print("[INFO]: step {}, loss {}".format(step, loss_v))
    
if __name__ == "__main__":
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    main()