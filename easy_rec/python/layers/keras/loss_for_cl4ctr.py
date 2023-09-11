import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from easy_rec.python.utils.shape_utils import get_shape_list


def contrastive(fea_cl1, fea_cl2):
    distance_sq = tf.reduce_sum(tf.square(tf.subtract(fea_cl1, fea_cl2)), axis=1)
    loss = tf.reduce_mean(distance_sq)
    return loss


def compute_uniformity_loss(fea_emb):
    frac = tf.matmul(fea_emb, tf.transpose(fea_emb, perm=[0, 2, 1]))
    norm = tf.norm(fea_emb, axis=2, keepdims=True)
    denom = tf.matmul(norm, tf.transpose(norm, perm=[0, 2, 1]))
    res = tf.div_no_nan(frac, denom)
    uniformity_loss = tf.reduce_mean(res)
    return uniformity_loss


def compute_alignment_loss(fea_emb):
    batch_size = get_shape_list(fea_emb)[0]
    indices = tf.where(tf.ones([tf.reduce_sum(batch_size), tf.reduce_sum(batch_size)]))
    row = tf.gather(tf.reshape(indices[:, 0], [-1]), tf.where(indices[:, 0] < indices[:, 1]))
    col = tf.gather(tf.reshape(indices[:, 1], [-1]), tf.where(indices[:, 0] < indices[:, 1]))
    row = tf.squeeze(row)
    col = tf.squeeze(col)
    x_row = tf.gather(fea_emb, row)
    x_col = tf.gather(fea_emb, col)
    distance_sq = tf.reduce_sum(tf.square(tf.subtract(x_row, x_col)), axis=2)
    alignment_loss = tf.reduce_mean(distance_sq)
    return alignment_loss


class LOSSCTR(Layer):
    def __init__(self, params, name='loss_ctr', **kwargs):
        super(LOSSCTR, self).__init__(name=name, **kwargs)
        self.cl_weight = params.get_or_default('cl_weight', 1)
        self.au_weight = params.get_or_default('au_weight', 0.01)

    def call(self, inputs, training=None, **kwargs):
        if training:
            # fea_cl1, fea_cl2, fea_emd = inputs
            fea_cl1, fea_cl2 = inputs
            # cl_align_loss = compute_alignment_loss(fea_emd)
            # cl_uniform_loss = compute_uniformity_loss(fea_emd)
            cl_loss = contrastive(fea_cl1, fea_cl2)
            # loss = cl_loss * self.cl_weight + (cl_align_loss + cl_uniform_loss) * self.au_weight
            loss_dict = kwargs['loss_dict']
            loss_dict['%s_cl_loss' % self.name] = cl_loss * self.cl_weight
            # loss_dict['%s_align_loss' % self.name] = cl_align_loss * self.au_weight
            # loss_dict['%s_uniform_loss' % self.name] = cl_uniform_loss * self.au_weight
        return 0
