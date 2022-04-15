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
import os
import argparse
import tensorflow as tf
import sparse_operation_kit as sok
import utils
from models import DLRM
from dataset import CriteoTsvReader
import time

def update_metrics_states(y_true, y_pred, metrics):
    y_pred = tf.nn.sigmoid(y_pred)
    for metric in metrics:
        if metric.name == "label_mean":
            metric.update_state(y_true)
        elif metric.name == "prediction_mean":
            metric.update_state(y_pred)
        else:
            metric.update_state(y_true, y_pred)

def train_loop_end(metrics, loss, val_loss, emb_opt, dense_opt, global_step):
    logs = {}
    for metric in metrics:
        logs[metric.name] = metric.result()
        metric.reset_states()
    for i, optimizer in enumerate([emb_opt, dense_opt]):
        lr_key = f'{type(optimizer).__name__}_{i}_learning_rate'
        if callable(optimizer.learning_rate):
            logs[lr_key] = optimizer.learning_rate(global_step)
        else:
            logs[lr_key] = optimizer.learning_rate
    logs["training_loss"] = loss if not hasattr(loss, "values") else loss.values[0]
    logs["validation_loss"] = val_loss if not hasattr(val_loss, "values") else val_loss.values[0]
    logs["global_step"] = global_step
    return logs
    
def main(args):
    comm_options = None

    if "mirrored" == args.distribute_strategy:
        avaiable_cuda_devices = ",".join([str(gpu_id) for gpu_id in range(args.gpu_num)])
        os.environ["CUDA_VISIBLE_DEVICES"] = avaiable_cuda_devices

        strategy = tf.distribute.MirroredStrategy()
        args.task_id = 0
    elif "multiworker" == args.distribute_strategy:
        args.task_id = int(os.getenv("OMPI_COMM_WORLD_RANK"))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.task_id)
        args.gpu_num = int(os.getenv("OMPI_COMM_WORLD_SIZE"))

        comm_options = tf.distribute.experimental.CommunicationOptions(
            bytes_per_pack=0,
            timeout_seconds=None,
            implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
        )

        import json
        port = 12345
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": {"worker": ["localhost" + ":" + str(port + i) 
                                    for i in range(args.gpu_num)]},
            "task": {"type": "worker", "index": args.task_id}
        })
        strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=comm_options)
    elif "horovod" == args.distribute_strategy:
        import horovod.tensorflow as hvd
        hvd.Init()

        args.task_id = hvd.local_rank()
        args.gpu_num = hvd.size()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.task_id)
        strategy = utils.NullStrategy()
    else:
        raise ValueError("Not supported distribute_strategy. "
                         f"Can only be one of ['mirrored', 'multiworker', 'horovod']"
                         f", but got {args.distribute_strategy}")

    with strategy.scope():
        if args.embedding_layer == "SOK":
            sok.Init(global_batch_size=args.global_batch_size)

        model = DLRM(vocab_size=args.vocab_size_list,
                     num_dense_features=args.num_dense_features,
                     embedding_layer=args.embedding_layer,
                     embedding_vec_size=args.embedding_vec_size,
                     bottom_stack_units=args.bottom_stack,
                     top_stack_units=args.top_stack,
                     TF_MP=args.TF_MP,
                     comm_options=comm_options)

        lr_callable = utils.get_lr_callable(global_batch_size=args.global_batch_size,
                                            decay_exp=args.decay_exp,
                                            learning_rate=args.learning_rate,
                                            warmup_steps=args.warmup_steps,
                                            decay_steps=args.decay_steps,
                                            decay_start_steps=args.decay_start_steps)

        embedding_optimizer = utils.get_optimizer(args.embedding_optimizer)
        embedding_optimizer.learning_rate = lr_callable
        dense_optimizer = utils.get_optimizer("Adam")

    batch_size = args.global_batch_size if args.distribute_strategy == "mirrored" \
                                        else args.global_batch_size // args.gpu_num
    if args.distribute_strategy != "mirrored":
        args.train_file_pattern = utils.shard_filenames(args.train_file_pattern, 
                                                        args.gpu_num, args.task_id)
        args.test_file_pattern = utils.shard_filenames(args.test_file_pattern,
                                                        args.gpu_num, args.task_id)

    train_dataset = CriteoTsvReader(file_pattern=args.train_file_pattern,
                                    num_dense_features=args.num_dense_features,
                                    vocab_sizes=args.vocab_size_list,
                                    batch_size=batch_size)
    val_dataset = CriteoTsvReader(file_pattern=args.test_file_pattern,
                                  num_dense_features=args.num_dense_features,
                                  vocab_sizes=args.vocab_size_list,
                                  batch_size=batch_size)
    
    distribute_dataset = (args.distribute_strategy == "mirrored" and args.gpu_num > 1)
    train_dataset = utils.get_distribute_dataset(train_dataset, strategy,
                                                 distribute_dataset=distribute_dataset)
    val_dataset = utils.get_distribute_dataset(val_dataset, strategy,
                                               distribute_dataset=distribute_dataset)
    val_dataset = iter(val_dataset)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, 
                        reduction=tf.keras.losses.Reduction.NONE)
    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss, global_batch_size=args.global_batch_size)

    metrics = [
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Mean("prediction_mean"),
        tf.keras.metrics.Mean("label_mean")
    ]
    metrics_threshold = {
        "auc": 0.8025
    }

    @tf.function
    def _train_step(features, labels, first_batch=False):
        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            loss = _replica_loss(labels, logits)

        emb_vars, other_vars = utils.split_embedding_variables_from_others(model)
        emb_grads, other_grads = tape.gradient(loss, [emb_vars, other_vars])

        with tf.control_dependencies([logits] + emb_grads):
            utils.apply_gradients(embedding_optimizer, emb_vars, emb_grads, 
                                args.embedding_layer == "SOK", 
                                aggregate_gradients = (not args.TF_MP))

            other_grads = utils.all_reduce(other_grads, combiner="sum", comm_options=comm_options)
            utils.apply_gradients(dense_optimizer, other_vars, other_grads,
                                False)

            if first_batch:
                utils.broadcast_variables(other_vars)
                utils.broadcast_variables(dense_optimizer.variables())

                if args.embedding_layer == "TF":
                    utils.broadcast_variables(emb_vars)
                    utils.broadcast_variables(embedding_optimizer.variables())

            total_loss = utils.all_reduce(loss, combiner="sum", comm_options=comm_options)
        return total_loss

    @tf.function
    def _val_step(features, labels, metrics):
        val_logits = model(features, training=False)
        val_loss = _replica_loss(labels, val_logits)
        val_loss = utils.all_reduce(val_loss, combiner="sum", comm_options=comm_options)

        labels = tf.identity(labels)
        val_logits = utils.all_gather(val_logits, axis=0, comm_options=comm_options)
        labels = utils.all_gather(labels, axis=0, comm_options=comm_options)
        
        return val_logits, labels, val_loss

    stopper = utils.EarlyStopper()

    begin_time = time.time()
    start_time = begin_time
    for i, (features, labels) in enumerate(train_dataset):
        if i >= args.train_steps:
            break
        if stopper.should_stop():
            print(stopper.stop_reason)
            break

        total_loss = strategy.run(_train_step, args=(features, labels, i == 0))

        if i % args.validation_interval == 0 and i != 0:
            val_features, val_labels = next(val_dataset)
            val_logits, val_labels, val_loss =\
                strategy.run(_val_step, args=(val_features, val_labels, metrics))

            if hasattr(val_labels, "values"):
                val_labels = val_labels.values[0]
                val_logits = val_logits.values[0]

            update_metrics_states(y_true=val_labels, y_pred=val_logits, metrics=metrics)
            val_logs = train_loop_end(metrics, total_loss, val_loss, embedding_optimizer, 
                                    dense_optimizer, global_step=i)

            elapsed_time = time.time() - begin_time
            steps_sec = args.validation_interval / elapsed_time
            utils.show_logs(val_logs, strategy, elapsed_time, steps_sec, metrics_threshold, stopper)
            begin_time = time.time()

    end_time = time.time()
    if args.task_id == 0:
        print(f"With {args.distribute_strategy} + {args.embedding_layer} embedding layer, "
              f"on {args.gpu_num} GPUs, and global_batch_size is {args.global_batch_size}, "
              f"it takes {end_time - start_time} seconds to "
              f"finish {args.train_steps} steps training for DLRM.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--global_batch_size", type=int, required=True)
    parser.add_argument("--train_file_pattern", type=str, required=True)
    parser.add_argument("--test_file_pattern", type=str, required=True)
    parser.add_argument("--embedding_layer", type=str, choices=["TF", "SOK"], required=True)
    parser.add_argument("--TF_MP", type=int, choices=[0, 1], default=0, required=False,
                        help="a flag to denote whether TF's Embedding works in Model-Parallel"
                        ", default to False, means it is working on data-parallel")
    parser.add_argument("--embedding_vec_size", type=int, required=True)
    parser.add_argument("--embedding_optimizer", type=str, required=False, default='SGD')
    parser.add_argument("--bottom_stack", type=int, nargs="+", required=True)
    parser.add_argument("--top_stack", type=int, nargs="+", required=True)
    parser.add_argument("--distribute_strategy", type=str, 
                        choices=["mirrored", "multiworker", "horovod"],
                        required=True)
    parser.add_argument("--gpu_num", type=int, required=False, default=1)
    parser.add_argument("--decay_exp", type=int, required=False, default=2)
    parser.add_argument("--learning_rate", type=float, required=False, default=1.25)
    parser.add_argument("--warmup_steps", type=int, required=False, default=8000)
    parser.add_argument("--decay_steps", type=int, required=False, default=30000)
    parser.add_argument("--decay_start_steps", type=int, required=False, default=70000)
    parser.add_argument("--validation_interval", type=int, required=False, default=100)
    parser.add_argument("--train_steps", type=int, required=False, default=-1)

    args = parser.parse_args()

    args.vocab_size_list = [39884407, 39043, 17289, 7420, 20263, 
                            3, 7120, 1543, 63, 38532952, 2953546, 
                            403346, 10, 2208, 11938, 155, 4, 976, 
                            14, 39979772, 25641295, 39664985, 585935, 
                            12972, 108, 36]
    args.num_dense_features = 13
    args.train_steps = 4195155968 // args.global_batch_size if args.train_steps == -1 else args.train_steps
    args.TF_MP = True if 1 == args.TF_MP else False

    main(args)
    