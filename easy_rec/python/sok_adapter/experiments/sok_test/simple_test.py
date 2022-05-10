import tensorflow as tf

test = tf.constant([1, 1])
test = tf.add(test, test)

test = tf.nn.convolution(
    tf.constant(1, shape=(9, 9, 9), dtype=tf.float32),
    tf.constant(1, shape=(3, 3, 3), dtype=tf.float32),
    'SAME'
)

with tf.Session() as sess:
    print(sess.run(test))
