import os

import tensorflow as tf

Rootdir = os.path.abspath(os.path.dirname(os.getcwd()))

TensorBoarddir = Rootdir + r"\TensorBoard\MNLR"
with tf.name_scope("input"):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
    tf.summary.histogram('input1', input1)

    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
    tf.summary.histogram('input2', input2)
with tf.name_scope("output"):
    output = tf.add_n([input1, input2], name="add")
    tf.summary.histogram('output', output)
merged = tf.summary.merge_all()
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    summary, _ = sess.run([merged, output])

    writer = tf.summary.FileWriter(TensorBoarddir, sess.graph)
    writer.add_summary(summary)
    writer.close()
