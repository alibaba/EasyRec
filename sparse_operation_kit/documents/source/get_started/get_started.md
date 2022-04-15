# Get Started With SparseOperationKit #
This document will walk you through simple demos to get you familiar with SparseOperationKit.

<div class="admonition note">
<p class="admonition-title">See also</p>
<p>For experts or more examples, please refer to Examples section</p>
</div>

<div class="admonition Important">
<p class="admonition-title">Important</p>
<p>In this document and other examples in SOK, you are assumed to be familiar with TensorFlow and other related tools.</p>
</div>

## Install SparseOperationKit ##
Please refer to the [*Installation* section](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/intro_link.html#installation) to install SparseOperationKit to your system.

## Import SparseOperationKit ##
```python
import sparse_operation_kit as sok
```
Now, SOK supports TensorFlow 1.15 and 2.x, and it will detect the version of TensorFlow automatically. And SOK API signatures are identical for TensorFlow 2.x and TensorFlow 1.15.

## TensorFlow 2.x ##

### Define a model with TensorFlow ###
The structure of this demo model is depicted in Fig 1.

<br><img src=../images/demo_model_structure.png></br>
<center><b>Fig 1. The structure of demo model</b></center>
<br>
<br>

#### Via Subclassing ####
You can create this demo model via subclassing `tf.keras.Model`. See also [TensorFlow's Docs](https://tensorflow.google.cn/guide/keras/custom_layers_and_models).

```python
import tensorflow as tf

class DemoModel(tf.keras.models.Model):
    def __init__(self,
                 max_vocabulary_size_per_gpu,
                 slot_num,
                 nnz_per_slot,
                 embedding_vector_size,
                 num_of_dense_layers,
                 **kwargs):
        super(DemoModel, self).__init__(**kwargs)

        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.slot_num = slot_num            # the number of feature-fileds per sample
        self.nnz_per_slot = nnz_per_slot    # the number of valid keys per feature-filed
        self.embedding_vector_size = embedding_vector_size
        self.num_of_dense_layers = num_of_dense_layers

        # this embedding layer will concatenate each key's embedding vector
        self.embedding_layer = sok.All2AllDenseEmbedding(
                    max_vocabulary_size_per_gpu=self.max_vocabulary_size_per_gpu,
                    embedding_vec_size=self.embedding_vector_size,
                    slot_num=self.slot_num,
                    nnz_per_slot=self.nnz_per_slot)

        self.dense_layers = list()
        for _ in range(self.num_of_dense_layers):
            self.layer = tf.keras.layers.Dense(units=1024, activation="relu")
            self.dense_layers.append(self.layer)

        self.out_layer = tf.keras.layers.Dense(units=1, activation=None)

    def call(self, inputs, training=True):
        # its shape is [batchsize, slot_num, nnz_per_slot, embedding_vector_size]
        emb_vector = self.embedding_layer(inputs, training=training)

        # reshape this tensor, so that it can be processed by Dense layer
        emb_vector = tf.reshape(emb_vector, shape=[-1, self.slot_num * self.nnz_per_slot * self.embedding_vector_size])

        hidden = emb_vector
        for layer in self.dense_layers:
            hidden = layer(hidden)

        logit = self.out_layer(hidden)
        return logit
```

#### Via Functional API ####
You can also create this demo model via `TensorFlow Functional API`. See [TensorFlow's Docs](https://tensorflow.google.cn/guide/keras/functional).

```python
import tensorflow as tf

def create_DemoModel(max_vocabulary_size_per_gpu,
                     slot_num,
                     nnz_per_slot,
                     embedding_vector_size,
                     num_of_dense_layers):
    # config the placeholder for embedding layer
    input_tensor = tf.keras.Input(
                type_spec=tf.TensorSpec(shape=(None, slot_num, nnz_per_slot), 
                dtype=tf.int64))

    # create embedding layer and produce embedding vector
    embedding_layer = sok.All2AllDenseEmbedding(
                max_vocabulary_size_per_gpu=max_vocabulary_size_per_gpu,
                embedding_vec_size=embedding_vector_size,
                slot_num=slot_num,
                nnz_per_slot=nnz_per_slot)
    embedding = embedding_layer(input_tensor)

    # create dense layers and produce logit
    embedding = tf.keras.layers.Reshape(
                target_shape=(slot_num * nnz_per_slot * embedding_vector_size,))(embedding)
    
    hidden = embedding
    for _ in range(num_of_dense_layers):
        hidden = tf.keras.layers.Dense(units=1024, activation="relu")(hidden)
    logit = tf.keras.layers.Dense(units=1, activation=None)

    model = tf.keras.Model(inputs=input_tensor, outputs=logit)
    return model
```

### Use SparseOperationKit with tf.distribute.Strategy ###
SparseOperationKit is compatible with `tf.distribute.Strategy`. More specificly, `tf.distribute.MirroredStrategy` and `tf.distribute.MultiWorkerMirroredStrategy`.

#### with tf.distribute.MirroredStrategy ####
Documents for [tf.distribute.MirroredStrategy](https://tensorflow.google.cn/api_docs/python/tf/distribute/MirroredStrategy?hl=en). `tf.distribute.MirroredStrategy` is a tool to support data-parallel synchronized training in single machine, where there exists multiple GPUs.

<div class="admonition Caution">
<p class="admonition-title">Caution</p>
<p>The programming model for MirroredStrategy is single-process & multi-threads. But due to the GIL in CPython interpreter, it is hard to fully leverage all available CPU cores, which might impact the end-to-end training / inference performance. Therefore, MirroredStrategy is not recommended for multiple GPUs synchronized training.</p>
</div>

***create MirroredStrategy***
```python
strategy = tf.distribute.MirroredStrategy()
```
<div class="admonition Tip">
<p class="admonition-title">Tip</p>
<p>By default, MirroredStrategy will use all available GPUs in one machine. If you want to specify how many GPUs are used or which GPUs are used for synchronized training, please set `CUDA_VISIBLE_DEVICES` or `tf.config.set_visible_devices`.</p>
</div>

***create model instance under MirroredStrategy.scope***
```python
global_batch_size = 65536
use_tf_opt = True

with strategy.scope():
    sok.Init(global_batch_size=global_batch_size)

    model = DemoModel(
        max_vocabulary_size_per_gpu=1024,
        slot_num=10,
        nnz_per_slot=5,
        embedding_vector_size=16,
        num_of_dense_layers=7)

    if not use_tf_opt:
        emb_opt = sok.optimizers.Adam(learning_rate=0.1)
    else:
        emb_opt = tf.keras.optimizers.Adam(learning_rate=0.1)

    dense_opt = tf.keras.optimizers.Adam(learning_rate=0.1)
```
For a DNN model built with SOK, `sok.Init` must be used to conduct initilizations. Please see [its API document](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/init.html#module-sparse_operation_kit.core.initialize).

***define training step***
```python
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
def _replica_loss(labels, logits):
    loss = loss_fn(labels, logits)
    return tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

@tf.function
def _train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = _replica_loss(lables, logits)
    emb_var, other_var = sok.split_embedding_variable_from_others(model.trainable_variables)
    grads, emb_grads = tape.gradient(loss, [other_var, emb_var])
    if use_tf_opt:
        with sok.OptimizerScope(emb_var):
            emb_opt.apply_gradients(zip(emb_grads, emb_var),
                                    experimental_aggregate_gradients=False)
    else:
        emb_opt.apply_gradients(zip(emb_grads, emb_var),
                                experimental_aggregate_gradients=False)
    dense_opt.apply_gradients(zip(grads, other_var))
    return loss
```

If you are using native TensorFlow optimizers, such as `tf.keras.optimizers.Adam`, then `sok.OptimizerScope` must be used. Please see [its API document](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/utils/opt_scope.html#sparseoperationkit-optimizer-scope).

***start training***
```python
dataset = ...

for i, (inputs, labels) in enumerate(dataset):
    replica_loss = strategy.run(_train_step, args=(inputs, labels))
    total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, replica_loss, axis=None)
    print("[SOK INFO]: Iteration: {}, loss: {}".format(i, total_loss))
```

After these steps, the `DemoModel` will be successfully trained.

#### With tf.distribute.MultiWorkerMirroredStrategy ####
Documents for [tf.distribute.MultiWorkerMirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy). `tf.distribute.MultiWorkerMirroredStrategy` is a tool to support data-parallel synchronized training in multiple machines, where there exists multiple GPUs in each machine.

<div class="admonition Caution">
<p class="admonition-title">Caution</p>
<p>The programming model for MultiWorkerMirroredStrategy is multi-processes & multi-threads. Each process owns multi-threads and controls all available GPUs in single machine. Due to the GIL in CPython interpreter, it is hard to fully leverage all available CPU cores in each machine, which might impact the end-to-end training / inference performance. Therefore, it is recommended to use multiple processes in each machine, and each process controls one GPU.</p>
</div>

<div class="admonition Important">
<p class="admonition-title">Important</p>
<p>By default, MultiWorkerMirroredStrategy will use all available GPUs in each process. Please set `CUDA_VISIBLE_DEVICES` or `tf.config.set_visible_devices` for each process to make each process controls different GPU.</p>
</div>

***create MultiWorkerMirroredStrategy***
```python
import os, json

worker_num = 8 # how many GPUs are used
task_id = 0    # this process controls which GPU

os.environ["CUDA_VISIBLE_DEVICES"] = str(task_id) # this procecss only controls this GPU

port = 12345 # could be arbitrary unused port on this machine
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {"worker": ["localhost:" + str(port + i)
                            for i in range(worker_num)]},
    "task": {"type": "worker", "index": task_id}
})
strategy = tf.distribute.MultiWorkerMirroredStrategy()
```

***Other Steps***<br>
The steps ***create model instance under MultiWorkerMirroredStrategy.scope***, ***define training step*** and ***start training*** are the same as which are described in [with tf.distribute.MirroredStrategy](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/get_started/get_started.html#with-tf-distribute-mirroredstrategy). Please check that section.

***launch training program***<br>
Because multiple CPU processes are used in each machine for synchronized training, therefore `MPI` can be used to launch this program. For example:
```shell
$ mpiexec -np 8 [mpi-args] python3 main.py [python-args]
```

### Use SparseOperationKit with Horovod ###
SparseOperationKit is also compatible with Horovod which is similar to `tf.distribute.MultiWorkerMirroredStrategy`.

***initialize horovod for tensorflow***
```python
import horovod.tensorflow as hvd
hvd.init()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank()) # this process only controls one GPU
```

***create model instance***
```python
global_batch_size = 65536
use_tf_opt = True

sok.Init(global_batch_size=global_batch_size)

model = DemoModel(max_vocabulary_size_per_gpu=1024,
                  slot_num=10,
                  nnz_per_slot=5,
                  embedding_vector_size=16,
                  num_of_dense_layers=7)

if not use_tf_opt:
    emb_opt = sok.optimizers.Adam(learning_rate=0.1)
else:
    emb_opt = tf.keras.optimizers.Adam(learning_rate=0.1)

dense_opt = tf.keras.optimizers.Adam(learning_rate=0.1)
```
For a DNN model built with SOK, `sok.Init()` must be called to conduct initializations. Please see [its API document](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/init.html#module-sparse_operation_kit.core.initialize).

***define training step***
```python
loss_fn = tf.keras.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
def _replica_loss(labels, logits):
    loss = loss_fn(labels, logits)
    return tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

@tf.function
def _train_step(inputs, labels, first_batch):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = _replica_loss(labels, logits)
    emb_var, other_var = sok.split_embedding_variable_from_others(model.trainable_variables)
    emb_grads, other_grads = tape.gradient(loss, [emb_var, other_var])
    if use_tf_opt:
        with sok.OptimizerScope(emb_var):
            emb_opt.apply_gradients(zip(emb_grads, emb_var),
                                    experimental_aggregate_gradients=False)
    else:
        emb_opt.apply_gradients(zip(emb_grads, emb_var),
                                experimental_aggregate_gradients=False)
    
    other_grads = [hvd.allreduce(grads) for grads in other_grads]
    dense_opt.apply_gradients(zip(other_grads, other_var))

    if first_batch:
        hvd.broadcast_variables(other_var, root_rank=0)
        hvd.broadcast_variables(dense_opt.variables(), root_rank=0)

    return loss
```
If you are using native TensorFlow optimizers, such as `tf.keras.optimizers.Adam`, then `sok.OptimizerScope` must be used. Please see [its API document](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/utils/opt_scope.html#sparseoperationkit-optimizer-scope). 

***start training***
```python
dataset = ...

for i, (inputs, labels) in enumerate(dataset):
    replica_loss = _train_step(inputs, labels, 0 == i)
    total_loss = hvd.allreduce(replica_loss)
    print("[SOK INFO]: Iteration: {}, loss: {}".format(i, total_loss))
```

***launch training program***<br>
You can use `horovodrun` or `mpiexec` to launch multiple processes in each machine for synchronized training. For example:
```shell
$ horovodrun -np 8 -H localhost:8 python3 main.py [python-args]
```

## TensorFlow 1.15 ##
SOK is also compatible with TensorFlow 1.15. Due to some restrictions in TF 1.15, only Horovod can be used as the communication tool.

### Using SparseOperationKit with Horovod ###
***initialize horovod for tensorflow***
```python
import horovod.tensorflow as hvd
hvd.init()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank()) # this process only controls one GPU
```

***create model instance***
```python
global_batch_size = 65536
use_tf_opt = True

sok_init_op = sok.Init(global_batch_size=global_batch_size)

model = DemoModel(max_vocabulary_size_per_gpu=1024,
                  slot_num=10,
                  nnz_per_slot=5,
                  embedding_vector_size=16,
                  num_of_dense_layers=7)

if not use_tf_opt:
    emb_opt = sok.optimizers.Adam(learning_rate=0.1)
else:
    emb_opt = tf.keras.optimizers.Adam(learning_rate=0.1)
dense_opt = tf.keras.optimizers.Adam(learning_rate=0.1)
```
For a DNN model built with SOK, `sok.Init` must be used to conduct initilizations. Please see [its API document](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/init.html#module-sparse_operation_kit.core.initialize).

***define training step***
```python
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
def _replica_loss(labels, logits):
    loss = loss_fn(labels, logits)
    return tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

def train_step(inputs, labels, training):
    logits = model(inputs, training=training)
    loss = _replica_loss(labels, logit)
    emb_var, other_var = sok.split_embedding_variable_from_others(model.trainable_variables)
    grads = tf.gradients(loss, emb_var + other_var, colocate_gradients_with_ops=True)
    emb_grads, other_grads = grads[:len(emb_var)], grads[len(emb_var):]

    if use_tf_opt:
        with sok.OptimizerScope(emb_var):
            emb_train_op = emb_opt.apply_gradients(zip(emb_grads, emb_var))
    else:
        emb_train_op = emb_opt.apply_gradients(zip(emb_grads, emb_var))

    other_grads = [hvd.allreduce(grad) for grad in other_grads]
    other_train_op = dense_opt.apply_gradients(zip(other_grads, other_var))

    with tf.control_dependencies([emb_train_op, other_train_op]):
        total_loss = hvd.reduce(loss)
        total_loss = tf.identity(total_loss)
        
        return total_loss
```
If you are using native TensorFlow optimizers, such as `tf.keras.optimizers.Adam`, then `sok.OptimizerScope` must be used. Please see [its API document](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/api/utils/opt_scope.html#sparseoperationkit-optimizer-scope).

***start training***
```python
dataset = ...

loss = train_step(inputs, labels)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(sok_init_op)
    sess.run(init_op)
    
    for step in range(iterations):
        loss_v = sess.run(loss)
        print("[SOK INFO]: Iteration: {}, loss: {}".format(step, loss_v))
```
**Please be noted that `sok_init_op` must be the first step in `sess.run`, even before variables initialization.**

***launch training program***
You can use `horovodrun` or `mpiexec` to launch multiple processes in each machine for synchronized training. For example:
```shell
$ horovodrun -np 8 -H localhost:8 python main.py [args]
```