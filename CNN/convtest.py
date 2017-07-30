import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Utility.Normalize import mnormalize
from Utility.XlsReader import readxlsbycol

Rootdir = os.path.abspath(os.path.dirname(os.getcwd()))
Modeldir = Rootdir + r"\Models\CNN\CNN.ckpt"
# Modeldir = Rootdir + r"\Models\CNN_Best\CNN.ckpt"
Datadir = Rootdir + r"\DataSet\HeatPredictionv2.xlsx"
TensorBoarddir = Rootdir + r"\TensorBoard\CNN"

Data_Sheet = "Sheet1"

xsina_train = []
xsina_test = []
for i in range(1, 17):
    xsina = readxlsbycol(Datadir, "Sheet2", i)[1:22]
    xsina = mnormalize(xsina)
    xsina_train.append(xsina)
for i in range(17, 20):
    xsina = readxlsbycol(Datadir, "Sheet2", i)[1:22]
    xsina = mnormalize(xsina)
    xsina_test.append(xsina)
xsina_train = np.array(xsina_train).T
# xsina_train = np.transpose(xsina_train)
# [16,21]
xsina_test = np.array(xsina_test).T
xsina_test = np.transpose(xsina_test)
# 3*21
# print(np.shape(xsina_test))

x = tf.unstack(xsina_train)
with tf.Session() as sess:
    x_unstack = tf.unstack(xsina_test, axis=1)
    # 21*(16,1)
    print(x_unstack)
    xt = tf.expand_dims(x_unstack[0], 1)
    xt = tf.expand_dims(xt, 0)
    # xt = tf.transpose(xt)
    # xt = tf.expand_dims(xt, 2)
    print(xt)
    # x = x_unstack[16, 1, 1]
    # for i in range(xsina_train.shape[0]):
    #     l = [x[i] for x in xsina_train]
    #     print(l)
    w = tf.get_variable('weights', [8, 1, 1, xsina_train.shape[1]],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
    w_unstack = tf.unstack(w, axis=3)
    b = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0.1))

    w2 = tf.get_variable('weights2', [6, 1, 1, xsina_train.shape[1]],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
    w2_unstack = tf.unstack(w2, axis=3)
    b2 = tf.get_variable('biases2', [1], initializer=tf.constant_initializer(0.1))

    init = tf.global_variables_initializer()
    sess.run(init)

    convsI = []
    for i in range(xsina_train.shape[1]):
        xconv = tf.expand_dims(x_unstack[i], 1)
        xconv = tf.expand_dims(xconv, 0)

        convI = tf.nn.conv1d(tf.cast(xconv, tf.float32), w_unstack[i], stride=1, padding="VALID", data_format="NHWC")
        convI = tf.nn.bias_add(convI, b)
        convI_actived = tf.nn.relu(convI)
        convI_actived = tf.expand_dims(convI_actived, 2)
        poolI = tf.nn.max_pool(convI_actived, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        convsI.append(poolI)
    # # # # [batch_size, sentence_length-ws+1, num_filters_A]
    # # #input[batch=1, channel=1, width=16]
    # # #w[filter_width=8, in_channels=1, out_channels=1]
    # # input[batch=1, 1, in_width=16, in_channels=1]
    # # w[1, filter_width=8, 1, 1]

    convsII = []
    for i in range(len(convsI)):
        convII = tf.squeeze(convsI[i], axis=-1)
        convII = tf.nn.conv1d(convII, w2_unstack[i], stride=1, padding="VALID", data_format="NHWC")
        convII = tf.nn.bias_add(convII, b2)
        convII_actived = tf.nn.relu(convII)
        convII_actived = tf.expand_dims(convII_actived, 2)
        poolII = tf.nn.max_pool(convII_actived, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding="VALID")
        convsII.append(poolII)
    print(tf.squeeze(convsII))
