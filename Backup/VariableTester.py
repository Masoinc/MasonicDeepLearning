import tensorflow as tf

if __name__ == '__main__':
    x = tf.Variable([[0.0], [0.0]])

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(x))

