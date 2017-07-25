import os

import tensorflow as tf

from Backup.MNLR.MultivariateNonLinearRegression import mnormalize
from Utility.XlsReader import readxlsbycol

Rootdir = os.path.abspath(os.path.dirname(os.getcwd()))
Modeldir = Rootdir + r"\Models\MNLR\MNLR.ckpt"
Datadir = Rootdir + r"\DataSet\HeatPrediction.xlsx"

test_sheet = "Sheet2"

saver = tf.train.import_meta_graph(Modeldir + r".meta")

train_data_size = 5

x_train = []
for i in range(1, 8):
    x = readxlsbycol(Datadir, test_sheet, i)[1:1 + train_data_size]
    x = mnormalize(x)
    x_train.append(x)
y_train = readxlsbycol(Datadir, test_sheet, 8)[1:1 + train_data_size]

X = tf.placeholder(tf.float32, shape=(None, 7), name='x_train')
# 输入参数 5*7(None * 7)

Y = tf.placeholder(tf.float32, shape=[None, 1], name='y_train')
# 输出结果 1*7(None * 1)

with tf.Session() as sess:
    saver.restore(sess, Modeldir)
    # sess.run(feed_dict={X: x_train, Y: y_train})
    # print(sess.run(tf.get_default_graph().get_tensor_by_name("y_predict:0")))
    sess.run(y_, feed_dict={X: [prev_seq]})