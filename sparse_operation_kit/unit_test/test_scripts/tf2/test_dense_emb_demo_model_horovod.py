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

import argparse
import os
import sys
sys.path.append('../../../') # where to find plugin
import sparse_operation_kit as sok
import utils

import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd

from dense_models import SOKDenseDemo, TfDenseDemo

def generate_dense_variables(input_channel, units):
    w, b = [], []
    for i in range(len(units)):
        if i == 0:
            w.append(tf.random.normal([input_channel, units[i]]))
        else:
            w.append(tf.random.normal([units[i-1], units[i]]))
        b.append(tf.random.normal([units[i]]))
    return w, b

def generate_vocabulary_table(vocabulary_size_per_gpu, embedding_vec_size, num_of_gpu):
    tensors = [tf.random.normal([vocabulary_size_per_gpu, embedding_vec_size]) for _ in range(num_of_gpu)]
    return tensors

def generate_data(args):
    dense_variables = generate_dense_variables(args.slot_num * args.nnz_per_slot * args.embedding_vec_size, 
                                               [args.num_dense_units for _ in range(args.num_dense_layers)])
    vocabulary_tensors = generate_vocabulary_table(args.max_vocabulary_size_per_gpu, args.embedding_vec_size, hvd.size())
    samples, labels = utils.generate_random_samples(num_of_samples=args.global_batch_size, 
                                                    vocabulary_size=args.max_vocabulary_size_per_gpu * hvd.size(), 
                                                    slot_num=args.slot_num, 
                                                    max_nnz=args.nnz_per_slot, 
                                                    use_sparse_mask=False)
    samples, labels = tf.convert_to_tensor(samples), tf.convert_to_tensor(labels)

    for i in range(args.num_dense_layers):
        # dense_variables[0] means weight, dense_variables[1] means bias
        dense_variables[0][i] = hvd.broadcast(dense_variables[0][i], root_rank=0)
        dense_variables[1][i] = hvd.broadcast(dense_variables[1][i], root_rank=0)
    for i in range(hvd.size()):
        vocabulary_tensors[i] = hvd.broadcast(vocabulary_tensors[i], root_rank=0)
    samples = hvd.broadcast(samples, root_rank=0)
    labels = hvd.broadcast(labels, root_rank=0)

    return dense_variables, vocabulary_tensors, samples, labels

def run_sok_model(args, dense_variables, vocabulary_tensors, samples, labels):
    # split sample and labels
    assert(args.global_batch_size % hvd.size() == 0)
    local_batch_size = args.global_batch_size // hvd.size()
    local_id = hvd.local_rank()
    samples = samples[local_id*local_batch_size : (local_id+1)*local_batch_size]
    labels = labels[local_id*local_batch_size : (local_id+1)*local_batch_size]

    sok.Init(global_batch_size=args.global_batch_size)

    model = SOKDenseDemo(max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu, 
                         embedding_vec_size=args.embedding_vec_size, 
                         slot_num=args.slot_num, 
                         nnz_per_slot=args.nnz_per_slot, 
                         num_dense_layers=args.num_dense_layers, 
                         num_dense_units=args.num_dense_units)

    #model.build(input_shape=(local_batch_size, args.slot_num * args.nnz_per_slot * args.embedding_vec_size))
    model(samples, training=False)
    for i in range(args.num_dense_layers):
        model.dense_layers[i].trainable_variables[0].assign(dense_variables[0][i])
        model.dense_layers[i].trainable_variables[1].assign(dense_variables[1][i])

    sok_saver = sok.Saver()
    init_tensors = [tensor.numpy() for tensor in vocabulary_tensors]
    sok_saver.load_embedding_values(model.embedding_layer.embedding_variable, init_tensors)
    
    embedding_optimizer = utils.get_embedding_optimizer(args.optimizer)(learning_rate=0.1)
    dense_optimizer = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)
    if args.mixed_precision:
        embedding_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            embedding_optimizer, initial_scale=1024)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        _dtype = loss.dtype
        loss = tf.cast(loss, tf.float32)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=args.global_batch_size)
        return tf.cast(loss, dtype=_dtype)

    @tf.function
    def _train_step(inputs, labels, first_batch):
        with tf.GradientTape() as tape, tf.GradientTape() as emb_tape:
            logit = model(inputs, training=True)
            replica_loss = _replica_loss(labels, logit)
            if args.mixed_precision:
                _loss = embedding_optimizer.get_scaled_loss(replica_loss)
            else:
                _loss = replica_loss
        
        tape = hvd.DistributedGradientTape(tape)

        emb_variable, other_variable = sok.split_embedding_variable_from_others(model.trainable_variables)
        emb_grads = emb_tape.gradient(_loss, emb_variable)
        grads = tape.gradient(_loss, other_variable)
        if args.mixed_precision:
            emb_grads = embedding_optimizer.get_unscaled_gradients(emb_grads)
            grads = embedding_optimizer.get_unscaled_gradients(grads)

        if 'plugin' not in args.optimizer:
            with sok.OptimizerScope(emb_variable):
                embedding_optimizer.apply_gradients(zip(emb_grads, emb_variable),
                                                    experimental_aggregate_gradients=False)
        else:
            embedding_optimizer.apply_gradients(zip(emb_grads, emb_variable),
                                                experimental_aggregate_gradients=False)
        dense_optimizer.apply_gradients(zip(grads, other_variable))

        # Note: broadcast should be done after the first gradient step to ensure optimizer initialization.
        if first_batch:
            hvd.broadcast_variables(other_variable, root_rank=0)
            hvd.broadcast_variables(dense_optimizer.variables(), root_rank=0)

        return replica_loss
    
    loss_list = []
    for i in range(args.iter_num):
        loss = _train_step(samples, labels, i == 0)
        loss_list.append(loss)
        print("[INFO]: Iteration: {}, loss={}".format(i, loss))
    return loss_list

def run_tf_model(args, dense_variables, vocabulary_tensors, samples, labels):
    model = TfDenseDemo(global_batch_size=args.global_batch_size,
                        vocabulary_size=args.max_vocabulary_size_per_gpu * hvd.size(),
                        slot_num=args.slot_num,
                        nnz_per_slot=args.nnz_per_slot,
                        num_dense_layers=args.num_dense_layers,
                        num_dense_units=args.num_dense_units,
                        embedding_vec_size=args.embedding_vec_size)

    #model.build(input_shape=(args.global_batch_size, args.slot_num * args.nnz_per_slot * args.embedding_vec_size))
    model(samples, training=False)
    for i in range(args.num_dense_layers):
        model.dense_layers[i].trainable_variables[0].assign(dense_variables[0][i])
        model.dense_layers[i].trainable_variables[1].assign(dense_variables[1][i])

    vocabulary_table = tf.concat(vocabulary_tensors, axis=0)
    for i in range(hvd.size()):
        model.params.assign(vocabulary_table)

    optimizer = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)
    if args.mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, initial_scale=1024)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit = model(inputs, training=True)
            loss = loss_fn(labels, logit)
            if args.mixed_precision:
                _loss = optimizer.get_scaled_loss(loss)
            else:
                _loss = loss
        grads = tape.gradient(_loss, model.trainable_variables)
        if args.mixed_precision:
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    
    loss_list = []
    for i in range(args.iter_num):
        loss = _train_step(samples, labels)
        loss_list.append(loss)
        print("[INFO]: Iteration: {}, loss={}".format(i, loss))
    return loss_list

def get_args():
    parser = argparse.ArgumentParser(description='test demo model with horovod.')
    parser.add_argument('--iter_num', type=int,
                        help='the number of testing iterations.',
                        required=False, default=100)
    parser.add_argument('--max_vocabulary_size_per_gpu', type=int,
                        required=False, default=8192)
    parser.add_argument('--slot_num', type=int,
                        help='the number of feature fields',
                        required=False, default=100)
    parser.add_argument('--nnz_per_slot', type=int,
                        help='the number of keys in each slot',
                        required=False, default=10)
    parser.add_argument('--embedding_vec_size', type=int,
                        help='the dimention of embedding vector',
                        required=False, default=4)
    parser.add_argument('--global_batch_size', type=int, required=False, default=1024)
    parser.add_argument('--optimizer', type=str,
                        help="use what optimizer",
                        required=False, default='adam',
                        choices=['plugin_adam', 'adam', 'sgd'])
    parser.add_argument('--num_dense_layers', type=int, required=False, default=6)
    parser.add_argument('--num_dense_units', type=int, required=False, default=1024)
    parser.add_argument("--mixed_precision", type=int, choices=[0,1], default=0)
    args = parser.parse_args()
    args.mixed_precision = True if 1 == args.mixed_precision else False
    return args

if __name__ == '__main__':

    args = get_args()

    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    local_rank = os.getenv("OMPI_COMM_WORLD_RANK")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    hvd.init()

    dense_variables, vocabulary_tensors, samples, labels = generate_data(args)

    sok_loss_list = run_sok_model(args, dense_variables, vocabulary_tensors, samples, labels)
    # compute_average_loss
    for i in range(args.iter_num):
        sok_loss_list[i] = hvd.allreduce(sok_loss_list[i])

    if hvd.local_rank() == 0:
        tf_loss_list = run_tf_model(args, dense_variables, vocabulary_tensors, samples, labels)

    if hvd.local_rank() == 0:
        for i in range(args.iter_num):
            print('Iteration: {}, sok={}, tf={}'.format(i, sok_loss_list[i], tf_loss_list[i]))
