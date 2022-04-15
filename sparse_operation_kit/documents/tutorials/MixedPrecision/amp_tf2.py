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

def main():
    global_batch_size = 1024
    slot_num = 10
    nnz_per_slot = 5

    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)

    strategy = tf.distribute.MirroredStrategy()

    dataset = utility.get_dataset(global_batch_size, read_batchsize=global_batch_size)
    dataset = strategy.experimental_distribute_dataset(dataset)

    with strategy.scope():
        sok.Init(global_batch_size=global_batch_size)

        model = utility.SOKDenseDemo(max_vocabulary_size_per_gpu=1024,
                                     embedding_vec_size=8,
                                     slot_num=slot_num,
                                     nnz_per_slot=nnz_per_slot,
                                     num_dense_layers=0)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    def _replica_loss(labels, logits):
        labels = tf.cast(labels, logits.dtype)
        loss = loss_fn(labels, logits)
        dtype = loss.dtype
        loss = tf.cast(loss, tf.float32)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)
        return tf.cast(loss, dtype)

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit = model(inputs, training=True)
            loss = _replica_loss(labels, logit)
            scaled_loss = optimizer.get_scaled_loss(loss)
        emb_vars, other_vars =\
            sok.split_embedding_variable_from_others(model.trainable_variables)
        scaled_emb_grads, scaled_other_grads = tape.gradient(
                            scaled_loss, [emb_vars, other_vars])
        emb_grads = optimizer.get_unscaled_gradients(scaled_emb_grads)
        other_grads = optimizer.get_unscaled_gradients(scaled_other_grads)
        with sok.OptimizerScope(emb_vars):
            optimizer.apply_gradients(zip(emb_grads, emb_vars),
                        experimental_aggregate_gradients=False)
        optimizer.apply_gradients(zip(other_grads, other_vars))
        return loss

    for step, (inputs, labels) in enumerate(dataset):
        replica_loss = strategy.run(train_step, args=(inputs, labels))
        total_loss = strategy.reduce("sum", replica_loss, axis=None)
        print("[INFO]: step {}, loss {}".format(step, total_loss))

if __name__ == "__main__":
    main()