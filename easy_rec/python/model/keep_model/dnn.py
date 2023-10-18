import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Zeros, glorot_normal
from tensorflow.keras.regularizers import l2

def activation_layer(activation):
    if isinstance(activation, str):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer

class DNN(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None, need_ppnet=False, pp_size=0,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.need_ppnet = need_ppnet
        self.pp_size = pp_size
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        if self.need_ppnet:
            input_size -= self.pp_size
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]
        if self.need_ppnet:
            self.pp_gates = [DNN([size, size], activation='relu', use_bn=False,
                output_activation='sigmoid') for size in hidden_units[:-1]]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units) - 1)]

        if self.output_activation:
            self.activation_layers += [activation_layer(self.output_activation)]
        else:
            self.activation_layers += [activation_layer(self.activation)]

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs
        if self.need_ppnet:
            deep_input = inputs[..., self.pp_size:]
            pp_input = inputs[..., :self.pp_size]
            #deep_input_wo_bp = tf.stop_gradient(deep_input)
            #pp_input = tf.concat([pp_input, deep_input_wo_bp], axis=-1)

        for i in range(len(self.hidden_units)):
            if self.need_ppnet:
                deep_input = 2*self.pp_gates[i](pp_input) * deep_input
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            try:
                fc = self.activation_layers[i](fc, training=training)
            except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                print("make sure the activation function use training flag properly", e)
                fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
