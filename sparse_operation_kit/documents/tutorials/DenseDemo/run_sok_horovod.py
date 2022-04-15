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
from models import SOKDenseDemo
import argparse
import sys
sys.path.append("../")
import utility
from utility import sparse_operation_kit as sok
import nvtx
import horovod.tensorflow as hvd
import os

def main(args):
    # Initialize horovod
    hvd.init()

    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    # Generate local filename
    # Assume the dataset has been splited in advance
    local_file = args.data_filename_prefix + str(hvd.local_rank()) + ".file"

    # generate local batch size
    assert(args.global_batch_size % hvd.size() == 0)
    local_batch_size = args.global_batch_size // hvd.size()

    dataset = utility.TFDataset(filename=local_file,
                                batchsize=local_batch_size,
                                as_sparse_tensor=False,
                                repeat=1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Because there is no tensorflow distribute strategy, sok.Init() will call horovod to
    # broadcast nccl id and random seed, so it must be called after hvd.init()
    sok.Init(global_batch_size=args.global_batch_size)

    model = SOKDenseDemo(max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu,
                         embedding_vec_size=args.embedding_vec_size,
                         slot_num=args.slot_num,
                         nnz_per_slot=args.nnz_per_slot,
                         num_dense_layers=args.num_dense_layers)

    embedding_optimizer = utility.get_embedding_optimizer(args.optimizer)(learning_rate=0.1)
    dense_optimizer = utility.get_dense_optimizer(args.optimizer)(learning_rate=0.1)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss, global_batch_size=args.global_batch_size)

    @tf.function
    def _train_step(inputs, labels, first_batch):
        with tf.GradientTape() as tape, tf.GradientTape() as emb_tape:
            logit = model(inputs, training=True)
            replica_loss = _replica_loss(labels, logit)

        # Horovod: wrap tf.GradientTape with Horovod DistributedGradientTape
        tape = hvd.DistributedGradientTape(tape)

        # There is no need to wrap the emb_tape because the communication is done by sok
        # emb_tape = hvd.DistributedGradientTape(emb_tape)

        emb_variable, other_variable = sok.split_embedding_variable_from_others(model.trainable_variables)

        # type(emb_tape) here is hvd.DistributedGradientTape
        # type(tape) here is tf.GradientTape
        emb_grads = emb_tape.gradient(replica_loss, emb_variable)
        grads = tape.gradient(replica_loss, other_variable)

        if "plugin" not in args.optimizer:
            with sok.OptimizerScope(emb_variable):
                embedding_optimizer.apply_gradients(zip(emb_grads, emb_variable),
                                                    experimental_aggregate_gradients=False)
        else:
            embedding_optimizer.apply_gradients(zip(emb_grads, emb_variable),
                                                experimental_aggregate_gradients=False)
        dense_optimizer.apply_gradients(zip(grads, other_variable))

        # Note: broadcast should be done after the first gradient step to ensure optimizer has been initialized.
        # There is no need to broadcast emb_variable and embedding_optimizer, because the parallel mode inside 
        # sok is model parallel and the communication is down by sok itself.
        if first_batch:
            hvd.broadcast_variables(other_variable, root_rank=0)
            hvd.broadcast_variables(dense_optimizer.variables(), root_rank=0)

        return replica_loss

    for i, (inputs, labels) in enumerate(dataset):
        if args.stop_at_iter > 0 and i >= args.stop_at_iter:
            break

        rng = nvtx.start_range(message="Iteration_" + str(i), color="blue")

        total_loss = _train_step(inputs, labels, i == 0)

        nvtx.end_range(rng)
        print("[INFO]: Iteration: {}, loss={}".format(i, total_loss))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run DNN model with SparseOperationKit")

    parser.add_argument("--data_filename_prefix", type=str,
                        help="the filename prefix of training data",
                        required=True)
    parser.add_argument("--global_batch_size", type=int,
                        required=True)
    parser.add_argument("--max_vocabulary_size_per_gpu", type=int,
                        required=True)
    parser.add_argument("--slot_num", type=int, required=True,
                        help="the number of feature fields")
    parser.add_argument("--nnz_per_slot", type=int, required=True,
                        help="the number of keys in each slot")
    parser.add_argument("--num_dense_layers", type=int, required=True,
                        help="the number of fully connected layers in this DNN model")
    parser.add_argument("--embedding_vec_size", type=int, required=True,
                        help="the dimension of embedding vectors")
    parser.add_argument('--optimizer', type=str,
                        help="use what optimizer",
                        required=False, default='plugin_adam',
                        choices=['plugin_adam', 'adam', 'sgd'])
    parser.add_argument("--stop_at_iter", type=int, required=False,
                        help="early stop the process if iteration reachs this setting.",
                        default=-1)

    args = parser.parse_args()

    main(args)
