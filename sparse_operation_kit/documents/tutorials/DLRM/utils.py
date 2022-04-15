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
import sparse_operation_kit as sok
from models import SOKEmbedding
import os, glob

class EarlyStopper:
    def __init__(self):
        self._stop = False

    def set_stop(self, message):
        self._stop = True
        self._stop_reason = message

    @property
    def stop_reason(self):
        return self._stop_reason

    def should_stop(self):
        return self._stop


class WarmUpAndPolyDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate callable for the embeddings.
    Linear warmup on [0, warmup_steps] then
    Constant on [warmup_steps, decay_start_steps]
    And polynomial decay on [decay_start_steps, decay_start_steps + decay_steps].
    """

    def __init__(self,
                batch_size: int,
                decay_exp: float = 2.0,
                learning_rate: float = 40.0,
                warmup_steps: int = 8000,
                decay_steps: int = 12000,
                decay_start_steps: int = 10000):
        super(WarmUpAndPolyDecay, self).__init__()
        self.batch_size = batch_size
        self.decay_exp = decay_exp
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_start_steps = decay_start_steps

    def __call__(self, step):
        decay_exp = self.decay_exp
        learning_rate = self.learning_rate
        warmup_steps = self.warmup_steps
        decay_steps = self.decay_steps
        decay_start_steps = self.decay_start_steps

        scal = self.batch_size / 2048

        adj_lr = learning_rate * scal
        if warmup_steps == 0:
            return adj_lr

        warmup_lr = step / warmup_steps * adj_lr
        global_step = tf.cast(step, tf.float32)
        decay_steps = tf.cast(decay_steps, tf.float32)
        decay_start_step = tf.cast(decay_start_steps, tf.float32)
        warmup_lr = tf.cast(warmup_lr, tf.float32)

        steps_since_decay_start = global_step - decay_start_step
        already_decayed_steps = tf.minimum(steps_since_decay_start, decay_steps)
        decay_lr = adj_lr * (
            (decay_steps - already_decayed_steps) / decay_steps)**decay_exp
        decay_lr = tf.maximum(0.0001, decay_lr)

        lr = tf.where(
            global_step < warmup_steps, warmup_lr,
            tf.where(
                tf.logical_and(decay_steps > 0, global_step > decay_start_step),
                decay_lr, adj_lr))

        lr = tf.maximum(0.01, lr)
        return lr

    def get_config(self):
        return {
            'batch_size': self.batch_size,
            'decay_exp': self.decay_exp,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps,
            'decay_start_steps': self.decay_start_steps
        }


def get_optimizer(optimizer=None):
    if not optimizer:
        return tf.keras.optimizers.Adam()
    else:
        return tf.keras.optimizers.get(optimizer)

def get_lr_callable(global_batch_size,
                    decay_exp,
                    learning_rate,
                    warmup_steps,
                    decay_steps,
                    decay_start_steps):
    return WarmUpAndPolyDecay(
        batch_size=global_batch_size,
        decay_exp=decay_exp,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_start_steps=decay_start_steps)

class NullScope(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return False

class NullStrategy(object):
    def scope(self):
        return NullScope()

    def run(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def gather(self, tensor, axis):
        import horovod.tensorflow as hvd
        return hvd.allgather(tensor)

def shard_filenames(file_pattern, num_pipelines, pipeline_id):
    matching_files = glob.glob(file_pattern)
    matching_files.sort()

    nums_per_shard = len(matching_files) // num_pipelines

    return matching_files[pipeline_id * nums_per_shard : (pipeline_id + 1) * nums_per_shard]

def get_distribute_dataset(dataset, strategy, distribute_dataset=True):
    if isinstance(strategy, NullStrategy) or not distribute_dataset:
        return dataset()
    else:
        return strategy.distribute_datasets_from_function(
            lambda input_context: dataset(input_context),
            options=tf.distribute.InputOptions()
        )

def split_embedding_variables_from_others(model):
    if isinstance(model.embedding_layer, SOKEmbedding):
        return sok.split_embedding_variable_from_others(model.trainable_variables)
    else:
        dense_vars = []
        for layer in model.layers:
            if layer != model.embedding_layer:
                dense_vars.extend(layer.trainable_variables)
        return model.embedding_layer.trainable_variables, dense_vars

def all_reduce(tensors, combiner="sum", comm_options=None):
    if tf.distribute.has_strategy():
        replica_ctx = tf.distribute.get_replica_context()
        return replica_ctx.all_reduce(combiner, tensors, options=comm_options)
    else:
        import horovod.tensorflow as hvd
        return [hvd.allreduce(tensor) for tensor in tensors]

def all_gather(tensors, axis=0, comm_options=None):
    if tf.distribute.has_strategy():
        replica_ctx = tf.distribute.get_replica_context()
        return replica_ctx.all_gather(tensors, axis=axis, options=comm_options)
    else:
        import horovod.tensorflow as hvd
        return [hvd.allgather(tensor) for tensor in tensors]

def apply_gradients(optimizer, variables, grads, using_sok, aggregate_gradients=False):
    if using_sok:
        with sok.OptimizerScope(variables):
            optimizer.apply_gradients(zip(grads, variables),
                                      experimental_aggregate_gradients=False)
    else:
        optimizer.apply_gradients(zip(grads, variables),
                                  experimental_aggregate_gradients=aggregate_gradients)


def broadcast_variables(variables):
    if tf.distribute.has_strategy():
        return
    else:
        import horovod.tensorflow as hvd
        hvd.broadcast_variables(variables, root_rank=0)

def show_logs(logs, strategy, elapsed_time, steps_sec, metrics_threshold, stopper):
    for key, value in logs.items():
        if hasattr(value, "values"):
            logs[key] = value.values[0]
        if hasattr(value, "numpy"):
            logs[key] = value.numpy()

    def no_print():
        return

    def print_logs():
        print("-"*23, logs["global_step"], "-"*23)
        del logs["global_step"]
        for key, value in logs.items():
            print(f"{key}: {logs[key]}")
        print("elapsed_time:", elapsed_time)
        print("steps/sec:", steps_sec)
        print("-"*50)

    if isinstance(strategy, NullStrategy):
        import horovod.tensorflow as hvd
        if hvd.local_rank() != 0:
            no_print()
        else:
            print_logs()
    elif os.getenv("OMPI_COMM_WORLD_RANK"):
        rank = os.getenv("OMPI_COMM_WORLD_RANK")
        if int(rank) != 0:
            no_print()
        else:
            print_logs()
    else:
        print_logs()

    for key, value in metrics_threshold.items():
        if logs[key] >= value:
            stopper.set_stop(
                f"Metric {key}: {logs[key]} meets its "
                f"threshold {value}, stop training.")
            break
