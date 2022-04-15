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

"""
This script do cross-checking with TF using multiple dense embedding layers.
"""

import argparse

import sys, os
sys.path.append(os.path.abspath(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../../../"))) # where to find SOK
import sparse_operation_kit as sok
import tensorflow as tf

import numpy as np
import json
import pickle
import utils

sys.path.append(os.path.abspath(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../../../documents/tutorials/DenseDemo")))
from models import SOKDenseModel, TFDenseModel

sys.path.append(os.path.abspath(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../../../documents/tutorials")))
import utility

def test_sok_multi_dense_emb(args):
    comm_options = tf.distribute.experimental.CommunicationOptions(
        bytes_per_pack=0,
        timeout_seconds=None,
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
    )

    if args.worker_num == 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        port = 12345
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": {"worker": ["localhost" + ":" + str(port + i) 
                                    for i in range(args.worker_num)]},
            "task": {"type": "worker", "index": args.task_id}
        })
        strategy = tf.distribute.MultiWorkerMirroredStrategy(
                    communication_options=comm_options)

    replica_batch_size = args.global_batch_size // (args.worker_num * 1)

    dataset = utility.TFDataset(filename=args.file_prefix + str(args.task_id) + ".file",
                                batchsize=replica_batch_size,
                                as_sparse_tensor=False,
                                repeat=1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    dynamic_input = True if args.dynamic_input == 1 else False

    with strategy.scope():
        sok.Init(global_batch_size=args.global_batch_size)

        model = SOKDenseModel(max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu,
                              embedding_vec_size_list=args.embedding_vec_size_list,
                              slot_num_list=args.slot_num_list,
                              nnz_per_slot_list=[args.nnz_per_slot for _ in range(len(args.slot_num_list))],
                              num_dense_layers=args.num_dense_layers,
                              dynamic_input=dynamic_input)

        emb_opt = utils.get_embedding_optimizer(args.optimizer)(learning_rate=0.1)
        dense_opt = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)
        if args.mixed_precision:
            emb_opt = tf.keras.mixed_precision.LossScaleOptimizer(emb_opt, initial_scale=1024)

    # set initial value to embedding variables.
    sok_saver = sok.Saver()
    for i, layer in enumerate(model.embedding_layers):
        init_tensors = utils.get_ones_tensor(max_vocab_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                             embedding_vec_size=args.embedding_vec_size_list[i],
                                             num=args.worker_num)
        sok_saver.load_embedding_values(layer.embedding_variable, init_tensors)

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
            logit, all_vectors = model(inputs, training=True)
            loss = _replica_loss(labels, logit)
            if args.mixed_precision:
                _loss = emb_opt.get_scaled_loss(loss)
            else:
                _loss = loss
        emb_variable, other_variable = sok.split_embedding_variable_from_others(model.trainable_variables)
        grads, emb_grads = tape.gradient(_loss, [other_variable, emb_variable])
        if args.mixed_precision:
            grads = emb_opt.get_unscaled_gradients(grads)
            emb_grads = emb_opt.get_unscaled_gradients(emb_grads)
        
        if "plugin" not in args.optimizer:
            with sok.OptimizerScope(emb_variable):
                emb_opt.apply_gradients(zip(emb_grads, emb_variable),
                                                    experimental_aggregate_gradients=False)
        else:
            emb_opt.apply_gradients(zip(emb_grads, emb_variable),
                                                experimental_aggregate_gradients=False)
       
        with tf.control_dependencies(emb_grads):
            # mannually all-reduce dense gradients
            replica_context = tf.distribute.get_replica_context()
            grads = replica_context.all_reduce("sum", grads, 
                                                options=comm_options)
            dense_opt.apply_gradients(zip(grads, other_variable),
                                            experimental_aggregate_gradients=False)

            # manually all-reduce loss, it is ok, because replica_loss has already been used to 
            # update local variables.
            loss = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, loss,
                                              options=comm_options)
        return loss, all_vectors, logit

    # save its results
    sok_results = list()
    for i, (inputs, labels) in enumerate(dataset):
        if args.stop_iter >= 0 and i >= args.stop_iter:
            break

        total_loss, all_vectors, logit = strategy.run(_train_step, args=(inputs, labels))
        print("[INFO]: Iteration: {}, loss={}".format(i, total_loss))

        with tf.device("CPU:0"):
            sok_results.append(all_vectors)

    return sok_results


def test_tf_multi_dense_emb(args):
    dataset_filenames = [args.file_prefix + str(task_id) + ".file"
                         for task_id in range(args.worker_num)]

    samples_total = [list() for _ in range(args.dataset_iter_num)]
    labels_total = [list() for _ in range(args.dataset_iter_num)]
    replica_batch_size = args.global_batch_size // args.worker_num
    for worker_id in range(args.worker_num):
        samples, labels = utils.restore_from_file(dataset_filenames[worker_id])
        for i in range(args.dataset_iter_num):
            samples_total[i].extend(samples[i * replica_batch_size : (i + 1) * replica_batch_size])
            labels_total[i].extend(labels[i * replica_batch_size : (i + 1) * replica_batch_size])
    samples_total = np.concatenate(samples_total, axis=0)
    labels_total = np.concatenate(labels_total, axis=0)

    dataset = utils.tf_dataset(samples_total, labels_total,
                               batchsize=args.global_batch_size,
                               to_sparse_tensor=False,
                               repeat=1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    model = TFDenseModel(vocabulary_size=args.max_vocabulary_size_per_gpu * args.worker_num,
                         embedding_vec_size_list=args.embedding_vec_size_list,
                         slot_num_list=args.slot_num_list,
                         nnz_per_slot_list=[args.nnz_per_slot for _ in range(len(args.slot_num_list))],
                         num_dense_layers=args.num_dense_layers)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    if args.mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, initial_scale=1024)

    # set initial value to embedding variables
    for i, param in enumerate(model.embedding_params):
        init_tensors = utils.get_ones_tensor(max_vocab_size_per_gpu=args.max_vocabulary_size_per_gpu * args.worker_num,
                                            embedding_vec_size=args.embedding_vec_size_list[i],
                                            num=1)
        param.assign(init_tensors[0])

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit, all_vectors = model(inputs, training=True)
            loss = loss_fn(labels, logit)
            if args.mixed_precision:
                _loss = optimizer.get_scaled_loss(loss)
            else:
                _loss = loss
        grads = tape.gradient(_loss, model.trainable_variables)
        if args.mixed_precision:
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, all_vectors

    # save its results
    tf_results = list()
    for i, (inputs, labels) in enumerate(dataset):
        if args.stop_iter >= 0 and i >= args.stop_iter:
            break

        loss, all_vectors = _train_step(inputs, labels)
        print("[INFO]: Iteration: {}, loss={}".format(i, loss))

        with tf.device("CPU:0"):
            tf_results.append(all_vectors)
    return tf_results

def compare_sok_and_tf(args):
    sok_results = test_sok_multi_dense_emb(args)
    utils.save_to_file("./sok_results_" + str(args.task_id) + ".file", sok_results)

    # use these as a barrier
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()

    # only process-0 to do the cross-checking.
    if args.task_id != 0:
         return

    tf_results = test_tf_multi_dense_emb(args)

    all_sok_results_list = list()
    for i in range(args.worker_num):
        sok_results = utils.restore_from_file("./sok_results_" + str(i) + ".file")
        sok_results = tf.concat(sok_results, axis=0) # [iter-num, replica-bs, vectors]
        all_sok_results_list.append(sok_results)
    all_sok_results_list = tf.concat(all_sok_results_list, axis=1)
    all_sok_results_list = tf.split(all_sok_results_list, num_or_size_splits=len(tf_results), axis=0)
    all_sok_results_list = [tf.squeeze(item) for item in all_sok_results_list]

    if len(all_sok_results_list) != len(tf_results):
        raise ValueError("The length of sok results is not equal to that of tensorflow.")

    if args.dynamic_input:
        atol = 1e0
        rtol = 1e-2
    elif args.mixed_precision:
        atol = 1e-2
        rtol = 1e-2
    else:
        atol = 1e-4
        rtol = 1e-4
    for i, sok_vector in enumerate(all_sok_results_list):
        tf.debugging.assert_near(tf.reshape(sok_vector, 
                                            shape=[-1, tf.shape(sok_vector)[-1]]),
                                tf_results[i],
                                atol=atol,
                                rtol=rtol,
                                message=("the values is not consistent on Iteration: %d" %i))

    print("\n[INFO]: For multiple dense embedding layer: with MPI + MultiWorkerMirroredStrategy, the embedding"+\
          " vectors obtained from SOK and TF are consistent for %d iterations." 
          " With mixed_precision = %s"
          %(len(sok_results), args.mixed_precision))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run DNN model with SparseOperationKit")

    parser.add_argument("--file_prefix", type=str,
                        help="the file_prefix for each GPU.", required=True)
    parser.add_argument("--global_batch_size", type=int,
                        required=True)
    parser.add_argument("--max_vocabulary_size_per_gpu", type=int,
                        required=True)
    parser.add_argument("--slot_num_list", type=int, nargs="+", required=True,
                        help="the number of feature fields")
    parser.add_argument("--nnz_per_slot", type=int, required=True,
                        help="the number of keys in each slot")
    parser.add_argument("--num_dense_layers", type=int, required=True,
                        help="the number of fully connected layers in this DNN model")
    parser.add_argument("--embedding_vec_size_list", type=int, nargs="+", required=True,
                        help="the dimension of embedding vectors")
    parser.add_argument('--optimizer', type=str,
                        help="use what optimizer",
                        required=False, default='plugin_adam',
                        choices=['plugin_adam', 'adam', 'sgd'])
    parser.add_argument("--dataset_iter_num", type=int, required=True,
                        help="the iter num for MPI + SOK")
    parser.add_argument("--stop_iter", type=int, required=False, default=-1,
                        help="early stop at which iteration.")
    parser.add_argument("--dynamic_input", type=int, required=False, default=0, choices=[0, 1],
                        help="whether to use unique before dense_fprop. 1 means dynamic_input,"+\
                            "0 means static_input.")
    parser.add_argument("--mixed_precision", type=int, choices=[0, 1], default=0)

    args = parser.parse_args()

    args.dynamic_input = True if 1 == args.dynamic_input else False
    args.mixed_precision = True if 1 == args.mixed_precision else False

    size = os.getenv("OMPI_COMM_WORLD_SIZE")
    if size is None:
        raise RuntimeError("This app must be launched with mpi.")
    size = int(size)
    args.worker_num = size

    task_id = int(os.getenv("OMPI_COMM_WORLD_RANK"))

    args.task_id = task_id

    os.environ["CUDA_VISIBLE_DEVICES"] = str(task_id)

    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    compare_sok_and_tf(args)

    # use these as a barrier
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()
