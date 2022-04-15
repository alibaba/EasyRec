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

import numpy as np
import os, json
import pickle
import utils

from test_dense_emb_demo_model_single_worker import SOKDenseDemo, test_tf_dense_model, check_saved_embedding_variables

def test_sok_dense_demo(args, init_tensors, *random_samples):
    port = 12345
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {"worker": [args.ips[i] + ":" + str(port + i) for i in range(args.worker_num)]},
        "task": {"type": "worker", "index": args.task_id}
    })
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        sok.Init(global_batch_size=args.global_batch_size)

        sok_dense_demo = SOKDenseDemo(max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                      embedding_vec_size=args.embedding_vec_size,
                                      slot_num=args.slot_num,
                                      nnz_per_slot=args.nnz_per_slot,
                                      use_hashtable=args.use_hashtable)

        emb_opt = utils.get_embedding_optimizer(args.optimizer)(learning_rate=0.1)
        dense_opt = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)
        if args.mixed_precision:
            emb_opt = tf.keras.mixed_precision.LossScaleOptimizer(emb_opt, initial_scale=1024)

    sok_saver = sok.Saver()
    if 1 == args.restore_params:
        filepath = r"./embedding_variables"
        sok_saver.restore_from_file(sok_dense_demo.embedding_layer.embedding_variable, filepath)
    else:
        sok_saver.load_embedding_values(sok_dense_demo.embedding_layer.embedding_variable, init_tensors)

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
            logit, embedding_vector = sok_dense_demo(inputs, training=True)
            loss = _replica_loss(labels, logit)
            if args.mixed_precision:
                _loss = emb_opt.get_scaled_loss(loss)
            else:
                _loss = loss

        embedding_variables, other_variable = sok.split_embedding_variable_from_others(sok_dense_demo.trainable_variables)
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

    sok_results = list()

    def _dataset_fn(input_context):
        replica_batch_size = input_context.get_per_replica_batch_size(args.global_batch_size)
        dataset = utils.tf_dataset(*random_samples, batchsize=replica_batch_size, 
                                   to_sparse_tensor=False, repeat=1)
        return dataset

    dataset = strategy.distribute_datasets_from_function(_dataset_fn)

    for i, (input_tensors, replica_labels) in enumerate(dataset):
        print("-"*30, "step ", str(i), "-"*30)
        loss, embedding_vector = strategy.run(_train_step, args=(input_tensors, replica_labels))
        loss = strategy.reduce("sum", loss, axis=None)
        print("[INFO]: iteration {}, loss {}".format(i, loss))
        sok_results.append(embedding_vector)


    # save params to file.
    if 1 == args.save_params:
        filepath = r"./embedding_variables"
        utils.try_make_dirs(filepath, chief=(True if args.task_id == 0 else False))

        sok_saver.dump_to_file(sok_dense_demo.embedding_layer.embedding_variable, filepath)

    return sok_results, sok_dense_demo.embedding_layer.embedding_variable.values[0].m_var_name

def compare_dense_emb_sok_with_tf(args):
    if (args.global_batch_size % args.local_gpu_num != 0):
        raise ValueError("global_batch_size: %d is not divisible by local_gpu_num: %d"
                        %(args.global_batch_size, args.local_gpu_num))
    if (args.global_batch_size % args.worker_num != 0):
        raise ValueError("global_batch_size: %d is not divisible by worker_num: %d"
                        %(args.global_batch_size, args.worker_num))

    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    #each worker generate different dataset
    if args.generate_new_datas:
        if args.use_hashtable:
            vocabulary_size = args.local_gpu_num * args.max_vocabulary_size_per_gpu * args.worker_num
        else:
            vocabulary_size = args.max_vocabulary_size_per_gpu

        worker_batch_size = args.global_batch_size // args.worker_num
        random_samples_local = utils.generate_random_samples(num_of_samples=worker_batch_size * args.iter_num,
                                                             vocabulary_size=vocabulary_size,
                                                             slot_num=args.slot_num,
                                                             max_nnz=args.nnz_per_slot,
                                                             use_sparse_mask=False)
        utils.save_to_file(r"./random_samples_" + str(args.task_id) + r".file", *random_samples_local)
    else:
        random_samples_local = utils.restore_from_file(r"./random_samples_" + str(args.task_id) + r".file")

    if 0 == args.restore_params:
        # each worker generate same init tensors, because each worker will do the filtering by itself
        init_tensors = utils.get_ones_tensor(max_vocab_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                            embedding_vec_size=args.embedding_vec_size,
                                            num=args.local_gpu_num * args.worker_num)
    else:
        filepath = r"./embedding_variables"
        tf_values_filename = os.path.join(filepath, r"tf_variable.file")
        init_tensors = utils.restore_from_file(tf_values_filename)

    sok_results_local, embedding_variable_name = test_sok_dense_demo(args, init_tensors, *random_samples_local)
    # save the forward embedding vector from different worker to file
    utils.save_to_file(r"./sok_embedding_vectors_" + str(args.task_id) + r".file", *sok_results_local)

    # only 1 process needs to do tf computation
    if args.task_id != 0:
        return

    # aggregate dataset from different worker
    dataset_filenames = [r"./random_samples_" + str(task_id) + r".file"
                         for task_id in range(args.worker_num)]
    random_samples_total = [list() for _ in range(args.iter_num)]
    random_labels_total = [list() for _ in range(args.iter_num)]
    local_batch_size = args.global_batch_size // args.worker_num
    for worker_id in range(args.worker_num):
        samples, labels = utils.restore_from_file(dataset_filenames[worker_id])
        for i in range(args.iter_num):
            random_samples_total[i].extend(samples[i * local_batch_size : (i + 1) * local_batch_size])
            random_labels_total[i].extend(labels[i * local_batch_size : (i + 1) * local_batch_size])
    random_samples_total = np.concatenate(random_samples_total, axis=0)
    random_labels_total = np.concatenate(random_labels_total, axis=0)

    tf_results = test_tf_dense_model(args, init_tensors, random_samples_total, random_labels_total)

    # aggregate forward embedding vector from different worker
    sok_results_filenames = [r"./sok_embedding_vectors_" + str(task_id) + r".file"
                             for task_id in range(args.worker_num)]
    sok_results_total = list()
    for file_name in sok_results_filenames:
        sok_results_local = utils.restore_from_file(file_name)
        sok_results_total.append(sok_results_local)
    
    if (len(sok_results_total[0]) != len(tf_results)):
        raise ValueError("The length of results obtained from sok: %d is not equal to that obtained from TF: %d"
                         %(len(sok_results_total[0]), len(tf_results)))
    if (len(tf_results) != args.iter_num):
        raise ValueError("The length of embedding vectors: %d is not equal to iteration number: %d."
                         %(len(tf_results), args.iter_num))
    
    if 1 == args.restore_params or args.mixed_precision:
        tolerance = 1e-2
    else:
        tolerance = 1e-4

    for i in range(args.iter_num):
        if args.local_gpu_num != 1:
            sok_vector = tf.concat([tf.concat(sok_results_total[task_id][i].values, axis=0)
                                    for task_id in range(args.worker_num)], axis=0)
        else:
            sok_vector = tf.concat([sok_results_total[task_id][i]
                                    for task_id in range(args.worker_num)],
                                    axis=0)
        tf.debugging.assert_near(tf.reshape(sok_vector,
                                            shape=[-1, tf.shape(sok_vector)[-1]]),
                                tf_results[i],
                                atol=tolerance,
                                rtol=tolerance)

    print("\n[INFO]: For Dense Embedding Layer, with MultiWorkerMirroredStrategy, the embedding vectors "+\
          "obtained from sparse operation kit and TensorFlow are consistent for %d iterations"
          ", with mixed_precision = %s"
          %(args.iter_num, args.mixed_precision))

    if 1 == args.save_params:
        check_saved_embedding_variables(args, embedding_variable_name, 
                                        use_hashtable=args.use_hashtable, 
                                        gpu_num=args.worker_num * args.local_gpu_num,
                                        atol=tolerance, rtol=tolerance)

def get_task_id(ips):
    local_ip = utils.get_local_ip()
    for i in range(len(ips)):
        if ips[i] == local_ip:
            return i
    raise ValueError("Cannot find local_ip: %s in ips list: [%s]"
                     %(local_ip, ", ".join(ips)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test demo model with single worker.')
    parser.add_argument('--local_gpu_num', type=int,
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
    parser.add_argument('--ips', type=str, nargs="+",
                        help="the ip address of each worker.",
                        required=False, default="0.0.0.0")
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

    args = parser.parse_args()

    args.use_hashtable = True if 1 == args.use_hashtable else False
    args.mixed_precision = True if 1 == args.mixed_precision else False

    if not isinstance(args.ips, list):
        args.ips = [args.ips]

    args.worker_num = len(args.ips)
    if utils.all_ips_in_local(args.ips):
        processes = list()
        for task_id in range(args.worker_num):
            available_gpus = ",".join([str(args.local_gpu_num * task_id + i)
                                       for i in range(args.local_gpu_num)])
            print("[INFO]: on task: %d, its available GPUs are: %s" %(task_id, available_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = available_gpus
            process = utils.TestProcess(func=compare_dense_emb_sok_with_tf, task_id=task_id, arguments=args)
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
    else:
        args.task_id = get_task_id(args.ips)

        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in range(args.local_gpu_num)])

        compare_dense_emb_sok_with_tf(args)