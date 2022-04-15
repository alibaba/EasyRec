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
import sys,os,json
sys.path.append("../")
import utility
from utility import sparse_operation_kit as sok
import nvtx
import time
import numpy as np


def main(args, task_id):
    print(task_id)
    comm_options = tf.distribute.experimental.CommunicationOptions(
        bytes_per_pack=0,
        timeout_seconds=None,
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
    )

    if args.total_gpu_num == 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        port = 12345
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": {"worker": ["localhost" + ":" + str(port + i) 
                                    for i in range(args.worker_num)]},
            "task": {"type": "worker", "index": task_id}
        })
        strategy = tf.distribute.MultiWorkerMirroredStrategy(
            communication_options=comm_options)

    if args.data_splited:
        filename = args.data_filename + str(task_id) + ".file"
    else:
        filename = args.data_filename

    replica_batch_size = args.global_batch_size // (args.worker_num * 1)

    dataset = utility.TFDataset(filename=filename, 
                                batchsize=replica_batch_size,
                                as_sparse_tensor=False, 
                                repeat=1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    with strategy.scope():
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
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit = model(inputs, training=True)
            loss = _replica_loss(labels, logit)
        emb_variable, other_variable = sok.split_embedding_variable_from_others(model.trainable_variables)
        grads, emb_grads = tape.gradient(loss, [other_variable, emb_variable])
        if "plugin" not in args.optimizer:
            with sok.OptimizerScope(emb_variable):
                embedding_optimizer.apply_gradients(zip(emb_grads, emb_variable),
                                                    experimental_aggregate_gradients=False)
        else:
            embedding_optimizer.apply_gradients(zip(emb_grads, emb_variable),
                                                experimental_aggregate_gradients=False)
        
        # mannually all-reduce dense gradients
        replica_context = tf.distribute.get_replica_context()
        grads = replica_context.all_reduce("sum", grads, 
                                            options=comm_options)
        dense_optimizer.apply_gradients(zip(grads, other_variable),
                                        experimental_aggregate_gradients=False)

        # manually all-reduce loss, it is ok, because replica_loss has already been used to 
        # update local variables.
        loss = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, loss,
                                          options=comm_options)
        return loss

    time_arr = []
    for i, (inputs, labels) in enumerate(dataset):
        if args.stop_at_iter > 0 and i >= args.stop_at_iter:
            break

        rng = nvtx.start_range(message="Iteration_" + str(i), color="blue")

        start_time = time.time()
        total_loss = strategy.run(_train_step, args=(inputs, labels))
        time_arr.append(time.time()-start_time)
        
        nvtx.end_range(rng)
        if task_id == '0':
            print("[INFO]: Iteration: {}, loss={}".format(i, total_loss))
            
    if task_id=='0':            
        print("Average iteration time (except 1st iteration): ", np.mean(time_arr[1:]))

def set_affinity(rank):
    affinity_map = {0: list(range(48,64)) + list(range(176,192)),
                    1: list(range(48,64)) + list(range(176,192)),
                    2: list(range(16,32)) + list(range(144,160)),
                    3: list(range(16,32)) + list(range(144,160)),
                    4: list(range(112,128)) + list(range(240,256)),
                    5: list(range(112,128)) + list(range(240,256)),
                    6: list(range(80,96)) + list(range(208,224)),
                    7: list(range(80,96)) + list(range(208,224))}

    my_affinity = affinity_map[rank]
    import os
    os.sched_setaffinity(0, my_affinity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run DNN model with SparseOperationKit")

    parser.add_argument("--data_filename", type=str,
                        help="the filename of training datas",
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
    parser.add_argument("--data_splited", type=int, required=False,
                        default=0, choices=[0, 1],
                        help="it is a flag used to denotes whether the data is already splited."+\
                             "by default, it is set to 0, which means the data is not splited.")
    parser.add_argument("--dgx_a100", action='store_true', 
                        help='Set if a DGX A100 is being used. In this case, CPU affinity will be set for optimal performance.')
    
    args = parser.parse_args()

    size = int(os.getenv("OMPI_COMM_WORLD_SIZE"))
    args.worker_num = size
    args.total_gpu_num = size

    task_id = os.getenv("OMPI_COMM_WORLD_RANK")

    if args.dgx_a100==True:
        print("Setting CPU affinity for DGX A100. This will likely fail on a non DGX A100 machine...")
        set_affinity(task_id)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(task_id)
    main(args, task_id)
