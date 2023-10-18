import math
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.python.keras.layers import Layer


class SENETLayer(Layer):
    """SENETLayer used in FiBiNET.
      Input shape
        - tensor with shape: ``(batch_size,field_size*embedding_size)``.
      Output shape
        - tensor with shape: ``(batch_size,field_size*embedding_size)``.
      Arguments
        - **reduction_ratio** : Positive integer, dimensionality of the
         attention network output space.
        - **seed** : A Python integer to use as random seed.
        - **norm** : norm function to norm attention factor, l2 or softmax.
        - **last_activation** : last act function, relu or identity.
        - **field_size** : feature field number.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, field_size, norm='l2', last_activation='identity', reduction_ratio=3, seed=1024, **kwargs):
        self.field_size = field_size
        self.norm = norm
        self.last_activation = last_activation
        self.reduction_ratio = reduction_ratio
        self.seed = seed
        super(SENETLayer, self).__init__(**kwargs)

    def tensordot(self, x, y):
        return tf.tensordot(x, y, axes=(-1, 0))

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError('A `SENETLayer` layer should be called '
                             'on a 2D tensor.')

        # if not divisible, raise an exception
        if input_shape[1] % self.field_size != 0:
            raise ValueError(f'Embed input size must be divisible by field_size, got {input_shape[1]} and {self.field_size}')
        self.embedding_size = int(input_shape[1] / self.field_size)
        reduction_size = max(1, self.field_size // self.reduction_ratio)

        self.W_1 = self.add_weight(shape=(
            self.field_size, reduction_size),
            initializer=glorot_normal(seed=self.seed), name="W1",
            trainable=True)
        self.W_2 = self.add_weight(shape=(
            reduction_size, self.field_size),
            initializer=glorot_normal(seed=self.seed), name="W2",
            trainable=True)

        # zty note:Lambda layer may cause variable not automatically
        # added to the variable set for gradient calculation
        # DO NOT use Lambda layer if possible.
        #self.tensordot = tf.keras.layers.Lambda(
        #    lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        # Be sure to call this somewhere!
        super(SENETLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):

        if  K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        inputs = tf.reshape(inputs, [-1, self.field_size, self.embedding_size])
        Z = tf.reduce_mean(inputs, axis=-1)

        A_1 = tf.nn.relu(self.tensordot(Z, self.W_1))
        A_2 = self.tensordot(A_1, self.W_2)
        if self.last_activation == 'relu':
            A_2 = tf.nn.relu(A_2)

        A_2 = tf.nn.l2_normalize(A_2, axis=1) if self.norm == 'l2' else tf.nn.softmax(A_2, axis=1)
        A_2 = A_2 * math.sqrt(self.field_size)
        output = tf.multiply(inputs, tf.expand_dims(A_2, axis=2))
        output = tf.reshape(output, [-1, self.field_size * self.embedding_size])

        return output

    def compute_output_shape(self, input_shape):

        return input_shape

    def compute_mask(self, inputs, mask=None):
        return [None] * self.field_size

    def get_config(self, ):
        config = {'reduction_ratio': self.reduction_ratio, 'seed': self.seed}
        base_config = super(SENETLayer, self).get_config()
        base_config.update(config)
        return base_config
