# Mixed Precision #
Mixed precision is a way to make an application run faster and use less memory. It use both 16-bit and 32-bit floating-point types during model training.

SparseOperationKit follows TensorFlow's pattern to enable mixed precision training, see this [link](https://tensorflow.google.cn/guide/mixed_precision?hl=en).

## TensorFlow 2.x ##
This section illustrates how to enable mixed precision training in TF 2.x.

### enable mixed precision ###
To use mixed precision in your model training, you just need to add the following two lines of code to your training script and keep other parts untouched.
```python
policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)
```

### loss scaling ###
The `float16` data type has a narrow dynamic range compared to `float32`, then `float16` might lead model training to underflow or overflow problems. Loss scaling is a technique to avoid numeric underflow. 

TensorFlow provides an optimizer wrapper for loss scaling. 
```python
optimizer = tf.keras.optimizers.Adam() # could be other optimizers as well
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
```

In the training loop, this optimizer wrapper should be used to calculate the scaled loss value. After the backward propagation and before trainable parameters' updating, that wrapper should be used to get the correct gradients. Because during backward propagation, loss is scaled, therefore gradients is also scaled, and when updating, you need to scale it back to correct value.
```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
        scaled_loss = optimizer.get_scaled_loss(loss) # this API should be called to get scaled loss
    emb_variables, other_variables =\
         sok.split_embedding_variable_from_others(model.trainable_variables)
    scaled_emb_gradients, scaled_other_gradients = tape.gradient(scaled_loss, 
                                            [emb_variables, other_variables])
    # use this API to scale embedding gradients back to correct value
    emb_gradients = optimizer.get_unscaled_gradients(scaled_emb_gradients) 
    # use this API to scale other gradients back to correct value
    other_gradients = optimizer.get_unscaled_gradients(scaled_other_gradients) 
    optimizer.apply_gradients(zip(emb_gradients, emb_variables),
                              experimental_aggregate_gradients=False)
    optimizer.apply_gradients(zip(other_gradients, other_variables))
    return loss
```

### example codes ###
You can find mixed precision example using TensorFlow 2.x in [`sparse_operation_kit/documents/tutorials/MixedPrecision`](https://github.com/NVIDIA/HugeCTR/tree/master/sparse_operation_kit/documents/tutorials/MixedPrecision). And use following command to launch it.
```shell
$ cd sparse_operation_kit/documents/tutorials/MixedPrecision/
$ python amp_tf2.py
```


## TensorFlow 1.15 ##
This section illustrates how to enable mixed precision training in TF 1.15.

### enable mixed precision ###
To use mixed precision in your model training, you just need to add the following four lines of code to your training script and keep other parts untouched.
```python
from tensorflow.python.keras.engine import base_layer_utils
base_layer_utils.enable_v2_dtype_behavior()

policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
tf.keras.mixed_precision.experimental.set_policy(policy)
```

### loss scaling ###
Similarily, optimizer wrapper should be used for loss scaling.
```python
optimizer = tf.keras.optimizers.Adam() # could be other optimizers as well
optimizer = sok.tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
```
`sok.tf.keras.mixed_precision.LossScaleOptimizer` is an optimized wrapper based on `tf.keras.mixed_precision.experimental.LossScaleOptimizer`, and it is only available in TensorFlow 1.15. 

They use the same arguments, you can find the arguments [**here**](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/mixed_precision/experimental/LossScaleOptimizer?hl=en).

In training loop, this optimizer wrapper should be used to calculate the scaled loss value, and get the unscaled gradients before parameters' updating.
```python
def train_step(x, y):
    predictions = model(x)
    loss = loss_object(y, predictions)
    # use this API to get scaled loss value
    scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_gradients = tf.gradients(scaled_loss, model.trainable_variables)
    # distinguish embedding variables and other variabels
    # because they need different processing.
    emb_variables, other_variables =\
        sok.split_embedding_variable_from_others(model.trainable_variables)
    scaled_emb_gradients, scaled_other_gradients =\
        scaled_gradients[:len(emb_variables)], scaled_gradients[len(emb_variables):]
    # use this API to scale embedding gradients back to correct value
    emb_gradients = optimizer.get_unscaled_gradients(scaled_emb_gradients)
    # use this API to scale other gradients back to correct value
    other_gradients = optimizer.get_unscaled_gradients(scaled_other_gradients)
    other_gradients = [hvd.allreduce(grad) for grad in other_gradients]
    emb_train_op = optimizer.apply_gradients(zip(emb_gradients, emb_variables))
    other_train_op = optimizer.apply_gradients(zip(other_gradients, other_variables))

    with tf.control_dependencies([emb_train_op, other_train_op]):
        return tf.identify(loss)
```

### example codes ###
You can find mixed precision example using TensorFlow 1.15 in [`sparse_operation_kit/documents/tutorials/MixedPrecision`](https://github.com/NVIDIA/HugeCTR/tree/master/sparse_operation_kit/documents/tutorials/MixedPrecision). And use following command to launch it.
```shell
$ cd sparse_operation_kit/documents/tutorials/MixedPrecision/
$ mpiexec --allow-run-as-root -np <gpu-number> python amp_tf1.py
```