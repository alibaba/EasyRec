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

import pickle
import utils

from sparse_models import SOKDemo, TfDemo, create_SOKSparseDemo_model

def test_sok_demo(args, init_tensors, *random_samples):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        result = sok.Init(global_batch_size=args.global_batch_size)

        embedding_initializer = tf.keras.initializers.Ones() if args.use_tf_initializer else None

        if args.functional_api:
            plugin_demo = create_SOKSparseDemo_model(
                                combiner=args.combiner,
                                max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                slot_num=args.slot_num,
                                max_nnz=args.max_nnz,
                                embedding_vec_size=args.embedding_vec_size,
                                use_hashtable=args.use_hashtable)
        else:
            plugin_demo = SOKDemo(combiner=args.combiner, 
                                  max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                  slot_num=args.slot_num, max_nnz=args.max_nnz,
                                  embedding_vec_size=args.embedding_vec_size,
                                  use_hashtable=args.use_hashtable,
                                  key_dtype=args.key_dtype,
                                  embedding_initializer=embedding_initializer)

        emb_opt = utils.get_embedding_optimizer(args.optimizer)(learning_rate=0.1)
        dense_opt = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)
        if args.mixed_precision:
            emb_opt = tf.keras.mixed_precision.LossScaleOptimizer(emb_opt, initial_scale=1024)

    plugin_saver = sok.Saver()

    if (1 == args.restore_params): # restore from trained parameters
        filepath = r"./embedding_variables"
        plugin_saver.restore_from_file(plugin_demo.embedding_layer.embedding_variable, filepath)
    else: # initialize using randomized initial value
        if not args.use_tf_initializer and init_tensors:
            status = plugin_saver.load_embedding_values(plugin_demo.embedding_layer.embedding_variable, init_tensors)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        _dtype = loss.dtype
        loss = tf.cast(loss, tf.float32)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=args.global_batch_size)
        return tf.cast(loss, _dtype)

    @tf.function
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit, embedding_vector = plugin_demo(inputs, training=True)
            loss = _replica_loss(labels, logit)
            if args.mixed_precision:
                _loss = emb_opt.get_scaled_loss(loss)
            else:
                _loss = loss
        embedding_variables, other_variable = sok.split_embedding_variable_from_others(plugin_demo.trainable_variables)
        grads, emb_grads = tape.gradient(_loss, [other_variable, embedding_variables])
        if args.mixed_precision:
            grads = emb_opt.get_unscaled_gradients(grads)
            emb_grads = emb_opt.get_unscaled_gradients(emb_grads)

        with tf.control_dependencies([*emb_grads]):
            # in case NCCL runs concurrently via SOK and TF
            if 'plugin' not in args.optimizer:
                with sok.OptimizerScope(embedding_variables):
                    emb_opt.apply_gradients(zip(emb_grads, embedding_variables),
                                            experimental_aggregate_gradients=False)
            else:
                emb_opt.apply_gradients(zip(emb_grads, embedding_variables),
                                        experimental_aggregate_gradients=False)
            dense_opt.apply_gradients(zip(grads, other_variable))
            return loss, embedding_vector

    sok_results = list()

    def _dataset_fn(input_context):
        replica_batch_size = input_context.get_per_replica_batch_size(args.global_batch_size)
        dataset = utils.tf_dataset(*random_samples, batchsize=replica_batch_size, 
                                   to_sparse_tensor=True, repeat=1, args=args)
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        return dataset

    dataset = strategy.distribute_datasets_from_function(_dataset_fn)
    
    for i, (sparse_tensors, replica_labels) in enumerate(dataset):
        print("-" * 30, "step ", str(i), "-" * 30)
        loss, embedding_vector = strategy.run(_train_step, args=(sparse_tensors, replica_labels))
        loss = strategy.reduce("sum", loss, axis=None)
        print("[INFO]: iteration {}, loss {}".format(i, loss))
        sok_results.append(embedding_vector)
    
    # save params to file.
    if 1 == args.save_params:
        filepath = r"./embedding_variables/"
        utils.try_make_dirs(filepath)

        plugin_saver.dump_to_file(plugin_demo.embedding_layer.embedding_variable, filepath)

    return sok_results, plugin_demo.embedding_layer.embedding_variable.values[0].m_var_name

def test_tf_demo(args, init_tensors, *random_samples):
    dataset = utils.tf_dataset(*random_samples, batchsize=args.global_batch_size, to_sparse_tensor=True, repeat=1)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    tf_demo = TfDemo(init_tensors, args.combiner, args.global_batch_size, args.slot_num, args.embedding_vec_size)

    optimizer = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)
    if args.mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, initial_scale=1024)

    @tf.function
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit, embedding_vector = tf_demo(inputs, training=True)
            loss = loss_fn(labels, logit)
            if args.mixed_precision:
                _loss = optimizer.get_scaled_loss(loss)
            else:
                _loss = loss
        grads = tape.gradient(_loss, tf_demo.trainable_variables)
        if args.mixed_precision:
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, tf_demo.trainable_variables))
        return loss, embedding_vector

    tf_results = list()

    for i, (sparse_tensors, labels) in enumerate(dataset):
        print("-"*30, str(i), "-"*30)
        loss, embedding_vector = _train_step(sparse_tensors, labels)
        print("[INFO]: iteration {}, loss {}".format(i, loss))
        tf_results.append(embedding_vector)

    if not hasattr(args, "task_id"):
        # In single worker, which means MirroedStrategy is used.
        args.task_id = 0
    if 1 == args.save_params and args.task_id == 0:
        filepath = r"./embedding_variables/"
        utils.save_to_file(os.path.join(filepath, r"tf_variable.file"), 
                           tf_demo.params.numpy())

    return tf_results

def check_saved_embedding_variables(args, embedding_variable_name, use_hashtable=True, gpu_num=None,
                                    atol=1e-4, rtol=1e-4):
    filepath = r"./embedding_variables"
    
    sok_keys_filename = os.path.join(filepath, embedding_variable_name + r"_keys.file")
    element_type = "long long"
    if hasattr(args, "key_dtype"):
        element_type = "long long" if args.key_dtype == "int64" else "unsigned int"
    sok_keys = utils.read_binary_file(sok_keys_filename, element_type=element_type)
    sok_values_filename = os.path.join(filepath, embedding_variable_name + r"_values.file")
    sok_values = utils.read_binary_file(sok_values_filename, element_type="float")

    sorted_sok_keys, sorted_sok_values = utils.sort_embedding_variables_by_key(sok_keys, sok_values, 
                                                    embedding_vec_size=args.embedding_vec_size,
                                                    use_hashtable=use_hashtable, gpu_num=gpu_num)

    tf_values_filename = os.path.join(filepath, r"tf_variable.file")
    tf_values = utils.restore_from_file(tf_values_filename)
    valid_tf_values = utils.get_valid_tf_values(sorted_sok_keys, tf_values[0])

    import numpy as np
    atol, rtol = atol, rtol
    sorted_sok_values = np.reshape(sorted_sok_values, newshape=(sorted_sok_keys.size, args.embedding_vec_size))
    allclose = np.allclose(sorted_sok_values, valid_tf_values, atol=atol, rtol=rtol)
    if not allclose:
        raise ValueError(f"The Variable from SOK: \n{sorted_sok_values}, \nis not near to that from TF: \n{valid_tf_values}"
                         f" \n at atol: {atol}, rtol: {rtol}")
    print("[INFO]: the saved parameters are consistent between sparse operation kit and TensorFlow")


def compare_sok_with_tf(args):
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
                                                    max_nnz=args.max_nnz)
        utils.save_to_file(r"./random_samples.file", *random_samples)
    else:
        random_samples = utils.restore_from_file(r"./random_samples.file")

    if (1 == args.restore_params): # initialize using trained params
        filepath = r"./embedding_variables"

        # because we already checked the Variable consistency when saving.
        # so that here we can directly use TensorFlow Variable file to initialize
        # tf's variable.
        # FIXME: what if not all TensorFlow embedding vectors are used??
        tf_values_filename = os.path.join(filepath, r"tf_variable.file")
        init_tensors = utils.restore_from_file(tf_values_filename)

    else: # initialize using random initial value
        init_tensors = utils.get_ones_tensor(max_vocab_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                            embedding_vec_size=args.embedding_vec_size,
                                            num=args.gpu_num)

    sok_results, embedding_variable_name = test_sok_demo(args, init_tensors, *random_samples)
    tf_results = test_tf_demo(args, init_tensors, *random_samples)

    if (len(sok_results) != len(tf_results)):
        raise ValueError("The length of plugin results is not equal to that of tensorflow.")
    if (len(tf_results) != args.iter_num):
        raise ValueError("The length of embedding vectors: %d is not equal to iteration number: %d."
                         %(len(tf_results), args.iter_num))

    tolerance = 1e-4
    if args.mixed_precision:
        tolerance = 1e-3

    for i, sok_vector in enumerate(sok_results):
        if args.gpu_num != 1:
            sok_vector = tf.stack(sok_vector.values, axis=0)
        tf.debugging.assert_near(tf.reshape(sok_vector,
                                            shape=[-1, tf.shape(sok_vector)[-1]]),
                                tf_results[i],
                                atol=tolerance,
                                rtol=tolerance)
    print("\n[INFO]: With MirroredStrategy, the embedding vector obtained from " +\
          "sparse operation kit and tensorflow are consistent for %d iterations." 
          " With mixed_precision = %s, and key_dtype = %s, and use_tf_initializer = %s"
          %(args.iter_num, args.mixed_precision, args.key_dtype, args.use_tf_initializer))

    if (1 == args.save_params):
        check_saved_embedding_variables(args, embedding_variable_name, 
                use_hashtable=args.use_hashtable, gpu_num=args.gpu_num,
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
    parser.add_argument('--max_nnz', type=int,
                        help='the maximum number of keys in one slot',
                        required=False, default=1)
    parser.add_argument('--embedding_vec_size', type=int,
                        help='the dimention of embedding vector',
                        required=False, default=1)
    parser.add_argument('--combiner', type=str,
                        help='the combiner used to do reduction for sparse embedding layer. ' +\
                             'It is only respected in sparse embedding layer.',
                        required=False, default='mean', choices=['mean', 'sum'])
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
    args.mixed_precision = True if 1 == args.mixed_precision else False
    args.use_tf_initializer = True if 1 == args.use_tf_initializer else False
    args.functional_api = True if 1 == args.functional_api else False

    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in range(args.gpu_num)])

    compare_sok_with_tf(args)