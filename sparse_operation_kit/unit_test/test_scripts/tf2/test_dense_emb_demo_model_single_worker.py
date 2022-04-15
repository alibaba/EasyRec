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

import sys, os
sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), r"../../../")))
import sparse_operation_kit as sok
import tensorflow as tf

from test_sparse_emb_demo_model_single_worker import check_saved_embedding_variables

import pickle
import utils

from dense_models import SOKDenseDemo, TfDenseDemo, create_SOKDenseDemo_model

def test_sok_dense_demo(args, init_tensors, *random_samples):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        result = sok.Init(global_batch_size=args.global_batch_size)

        embedding_initializer = tf.keras.initializers.Ones() if args.use_tf_initializer else None

        if args.functional_api:
            sok_dense_demo = create_SOKDenseDemo_model(max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                                       embedding_vec_size=args.embedding_vec_size,
                                                       slot_num=args.slot_num,
                                                       nnz_per_slot=args.nnz_per_slot,
                                                       use_hashtable=args.use_hashtable)
        else:
            sok_dense_demo = SOKDenseDemo(max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                          embedding_vec_size=args.embedding_vec_size,
                                          slot_num=args.slot_num,
                                          nnz_per_slot=args.nnz_per_slot,
                                          use_hashtable=args.use_hashtable,
                                          key_dtype=args.key_dtype,
                                          embedding_initializer=embedding_initializer)
        emb_opt = utils.get_embedding_optimizer(args.optimizer)(learning_rate=0.1)
        dense_opt = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)
        if args.mixed_precision:
            # only one LossScaleOptimizer is needed, since it will not track trainable variables.
            emb_opt = tf.keras.mixed_precision.LossScaleOptimizer(emb_opt, initial_scale=1024)

    sok_saver = sok.Saver()

    if 1 == args.restore_params:
        filepath = r"./embedding_variables"
        sok_saver.restore_from_file(sok_dense_demo.embedding_layer.embedding_variable, filepath)
    else:
        if not args.use_tf_initializer:
            sok_saver.load_embedding_values(sok_dense_demo.embedding_layer.embedding_variable, init_tensors)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        _dtype = loss.dtype
        loss = tf.cast(loss, dtype=tf.float32)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=args.global_batch_size)
        return tf.cast(loss, dtype=_dtype)

    @tf.function
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit, embedding_vector = sok_dense_demo(inputs, training=True)
            loss = _replica_loss(labels, logit)
            if args.mixed_precision:
                _loss = emb_opt.get_scaled_loss(loss)
            else:
                _loss = loss
        embedding_variables, other_variable = sok.split_embedding_variable_from_others(
                                                    sok_dense_demo.trainable_variables)
        grads, emb_grads = tape.gradient(_loss, [other_variable, embedding_variables])
        if args.mixed_precision:
            grads = emb_opt.get_unscaled_gradients(grads)
            emb_grads = emb_opt.get_unscaled_gradients(emb_grads)
        if "plugin" not in args.optimizer:
            with sok.OptimizerScope(embedding_variables):
                emb_opt.apply_gradients(zip(emb_grads, embedding_variables),
                                        experimental_aggregate_gradients=False)
        else:
            emb_opt.apply_gradients(zip(emb_grads, embedding_variables),
                                    experimental_aggregate_gradients=False)
        dense_opt.apply_gradients(zip(grads, other_variable))
        return loss, embedding_vector

    def _dataset_fn(input_context):
        replica_batch_size = input_context.get_per_replica_batch_size(args.global_batch_size)
        dataset = utils.tf_dataset(*random_samples, batchsize=replica_batch_size, 
                                    to_sparse_tensor=False, repeat=1, args=args)
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        return dataset

    dataset = strategy.distribute_datasets_from_function(_dataset_fn)

    sok_results = list()
    for i, (input_tensors, replica_labels) in enumerate(dataset):
        print("-" * 30, "step ", str(i), "-" * 30)
        loss, embedding_vector = strategy.run(_train_step, args=(input_tensors, replica_labels))
        loss = strategy.reduce("sum", loss, axis=None)
        print("[INFO]: iteration {}, loss {}".format(i, loss))
        sok_results.append(embedding_vector)

    if 1 == args.save_params:
        filepath = r"./embedding_variables/"
        utils.try_make_dirs(filepath)

        sok_saver.dump_to_file(sok_dense_demo.embedding_layer.embedding_variable, filepath)

    return sok_results, sok_dense_demo.embedding_layer.embedding_variable.values[0].m_var_name

def test_tf_dense_model(args, init_tensors, *random_samples):
    dataset = utils.tf_dataset(*random_samples, batchsize=args.global_batch_size,
                                to_sparse_tensor=False, repeat=1)
    
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    tf_dense_demo = TfDenseDemo(init_tensors, args.global_batch_size, args.slot_num,
                                args.nnz_per_slot, args.embedding_vec_size)
    
    optimizer = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)
    if args.mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, initial_scale=1024)

    @tf.function
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit, embedding_vector = tf_dense_demo(inputs, training=True)
            loss = loss_fn(labels, logit)
            if args.mixed_precision:
                _loss = optimizer.get_scaled_loss(loss)
            else:
                _loss = loss
        grads = tape.gradient(_loss, tf_dense_demo.trainable_variables)
        if args.mixed_precision:
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, tf_dense_demo.trainable_variables))
        return loss, embedding_vector

    tf_results = list()

    for i, (input_tensors, labels) in enumerate(dataset):
        print("-"*30, str(i), "-"*30)
        loss, embedding_vector = _train_step(input_tensors, labels)
        print("[INFO]: iteration {}, loss {}".format(i, loss))
        tf_results.append(embedding_vector.numpy())

    if not hasattr(args, "task_id"):
        args.task_id = 0
    if 1 == args.save_params and args.task_id == 0:
        filepath = r"./embedding_variables/"
        utils.save_to_file(os.path.join(filepath, r"tf_variable.file"),
                           tf_dense_demo.params.numpy())

    return tf_results

def compare_dense_emb_sok_with_tf(args):
    if (args.global_batch_size % args.gpu_num != 0):
        raise ValueError("global_batch_size: %d is not divisible by gpu_num: %d"
                        %(args.global_batch_size, args.gpu_num))

    if args.use_hashtable:
        vocabulary_size = args.max_vocabulary_size_per_gpu * args.gpu_num
    else:
        vocabulary_size = args.max_vocabulary_size_per_gpu

    if args.generate_new_datas:
        random_samples = utils.generate_random_samples(num_of_samples=args.global_batch_size * args.iter_num,
                                                    vocabulary_size=vocabulary_size,
                                                    slot_num=args.slot_num,
                                                    max_nnz=args.nnz_per_slot,
                                                    use_sparse_mask=False)
        utils.save_to_file(r"./random_samples.file", *random_samples)
    else:
        random_samples = utils.restore_from_file(r"./random_samples.file")

    if 1 == args.restore_params:
        filepath = r"./embedding_variables"

        # because we already checked the Variable consistency when saving.
        # so that here we can directly use TensorFlow Variable file to initialize
        # tf's variable.
        # FIXME: what if not all TensorFlow embedding vectors are used??
        tf_values_filename = os.path.join(filepath, r"tf_variable.file")
        init_tensors = utils.restore_from_file(tf_values_filename)
    else:
        init_tensors = utils.get_ones_tensor(
                    max_vocab_size_per_gpu=args.max_vocabulary_size_per_gpu,
                    embedding_vec_size=args.embedding_vec_size,
                    num=args.gpu_num)

    sok_results, embedding_variable_name = test_sok_dense_demo(args, init_tensors, *random_samples)
    tf_results = test_tf_dense_model(args, init_tensors, *random_samples)

    if len(sok_results) != len(tf_results):
        raise ValueError("The length of sok results is not equal to that of tensorflow.")
    if len(sok_results) != args.iter_num:
        raise ValueError("The length of embedding vectors: %d is not equal to iteration number: %d."
                        %(len(sok_results), args.iter_num))

    if 1 == args.restore_params or args.mixed_precision:
        tolerance = 1e-2
    else:
        tolerance = 1e-4

    for i, sok_vector in enumerate(sok_results):
        if args.gpu_num != 1:
            sok_vector = tf.stack(sok_vector.values, axis=0)
        tf.debugging.assert_near(tf.reshape(sok_vector, 
                                            shape=[-1, tf.shape(sok_vector)[-1]]),
                                tf_results[i],
                                atol=tolerance,
                                rtol=tolerance,
                                message="the values is not consistent on Iteration: %d" %i)

    print("\n[INFO]: For Dense Embedding Layer: with MirroredStrategy, the embedding vector obtained from " +\
          "sparse operation kit and TensorFlow are consistent for %d iterations with mixed_precision = %s, "
          "and key_dtype = %s, and use_tf_initializer = %s" 
          %(args.iter_num, args.mixed_precision, args.key_dtype, args.use_tf_initializer))

    if 1 == args.save_params:
        check_saved_embedding_variables(args, embedding_variable_name, 
                                        use_hashtable=args.use_hashtable, 
                                        gpu_num=args.gpu_num,
                                        atol=tolerance, rtol=tolerance)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test demo model with single worker.')
    parser.add_argument('--gpu_num', type=int,
                        help='the number of GPUs used to do paralell training.',
                        required=False, default=8)
    parser.add_argument('--iter_num', type=int,
                        help='the number of testing iterations.',
                        required=False, default=100)
    parser.add_argument('--max_vocabulary_size_per_gpu', type=int,
                        required=False, default=128)
    parser.add_argument('--slot_num', type=int,
                        help='the number of feature fields',
                        required=False, default=1)
    parser.add_argument('--nnz_per_slot', type=int,
                        help='the number of keys in each slot',
                        required=False, default=1)
    parser.add_argument('--embedding_vec_size', type=int,
                        help='the dimention of embedding vector',
                        required=False, default=1)
    parser.add_argument('--global_batch_size', type=int, required=False, default=16)
    parser.add_argument('--optimizer', type=str,
                        help="use what optimizer",
                        required=False, default='plugin_adam',
                        choices=['plugin_adam', 'adam', 'sgd'])
    parser.add_argument('--generate_new_datas', type=int, choices=[0, 1],
                        help='whether to generate new random samples',
                        required=False, default=1)
    parser.add_argument('--save_params', type=int, choices=[0, 1],
                        help='whether to save the trained parameters.',
                        required=False, default=0)
    parser.add_argument('--restore_params', type=int, choices=[0, 1],
                        help='whether to restore from saved files. '+\
                             'By default, the testing program will generate random ' +\
                             'initial value to initialize trainable parameters '+\
                             'rather than restore trainable parameters from file.',
                        required=False, default=0)
    parser.add_argument("--use_hashtable", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mixed_precision", type=int, choices=[0, 1], default=0)
    parser.add_argument("--key_dtype", type=str, choices=['int64', 'uint32'], default='int64')
    parser.add_argument("--use_tf_initializer", type=int, choices=[0, 1], default=0)
    parser.add_argument("--functional_api", type=int, choices=[0, 1], default=0)

    args = parser.parse_args()

    args.use_hashtable = True if args.use_hashtable == 1 else False
    args.mixed_precision = True if args.mixed_precision == 1 else False
    args.use_tf_initializer = True if 1 == args.use_tf_initializer else False
    args.functional_api = True if 1 == args.functional_api else False

    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in range(args.gpu_num)])

    compare_dense_emb_sok_with_tf(args)