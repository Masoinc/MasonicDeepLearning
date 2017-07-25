import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from LSTM.LongShortTermMemory import getrsqured
from Utility.XlsReader import readxlsbycol, readxlsbyrow

Rootdir = os.path.abspath(os.path.dirname(os.getcwd()))
Modeldir = Rootdir + r"\Models\MNLR\MNLR.ckpt"
Datadir = Rootdir + r"\DataSet\HeatPrediction.xlsx"
TensorBoarddir = Rootdir + r"\TensorBoard\MNLR"

data_sheet = "Sheet1"

test_data_size = 5


# Min-Max 标准化
def mnormalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Z-Score 标准化
def znormalize(data):
    return (data - np.mean(data)) / np.std(data)


# 训练参数
train_step = 10000
w1_regularizer_rate = 0.01
w2_regularizer_rate = 0.01
hidden_layer_size = 20
v_amount = 7

# batch_size = 3
learning_rate = 0.0001
# 指数衰减学习率
global_step = tf.Variable(0, name="global_step")
learning_rate = \
    tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.9,
                               staircase=True)


# 标准化(无量纲化)数据
def get_data():
    x_train = []
    x_test = []
    for i in range(1, 8):
        x = readxlsbycol(Datadir, data_sheet, i)[1:25]
        x = mnormalize(x)
        x_train.append(x[:19])
        x_test.append(x[19:])
        # print(x)
        # x = [x]
        # x_total = np.concatenate([x_total, x], axis=0)
    x_train = np.array(x_train).T  # 列表转矩阵(7*19)
    x_test = np.array(x_test).T

    # print(x_total)
    # print(x_total[0])
    # xtest = np.expand_dims(x_train[0], axis=1).transpose()
    # print(xtest)
    y_train = readxlsbycol(Datadir, data_sheet, 8)[1:20]
    y_train = mnormalize(y_train)
    y_train = np.expand_dims(y_train, axis=1)
    return x_train, x_test, y_train


def get_data_nonnormailized():
    x_train = []
    x_test = []
    for i in range(1, 8):
        x = readxlsbycol(Datadir, data_sheet, i)[1:25]
        x_train.append(x[:19])
        x_test.append(x[19:])

    x_train = np.array(x_train).T  # 列表转矩阵(7*19)
    x_test = np.array(x_test).T

    return x_train, x_test


# print(y)
# print(x_total[1][1])
# x = readxlsbycol(Directory, Sheet, 1)[1:20]
# x = mnormalize(x)
# # x1 = np.expand_dims(x, axis=0).transpose().tolist() # shape=(7,1)
#
# # x = np.expand_dims(x, axis=1).transpose().tolist()  # shape=(1,7)
# x = np.expand_dims(x, axis=1).transpose()  # shape=(1,7)
#
# ''
# # x = [[38, 1, 0, 1, 23590.5714285714, 36.9848681194, 4]]
# y = readxlsbyrow(Directory, Sheet, 1)[8]
# y = np.expand_dims(y, axis=1)


# X1 = tf.transpose(tf.expand_dims(X1, axis=0, name='X1'))
X = tf.placeholder(tf.float32, shape=(None, v_amount), name='x_train')
# 输入参数 7*19(None * 7)

Y = tf.placeholder(tf.float32, shape=[None, 1], name='y_train')


# 输出结果 1*19(None * 1)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn():
    with tf.name_scope('nn_layer'):
        with tf.name_scope('weights1'):
            w1 = tf.Variable((tf.random_normal([v_amount, hidden_layer_size], stddev=1)), name='w1')
            variable_summaries(w1)
        with tf.name_scope('weights2'):
            w2 = tf.Variable(tf.random_normal([hidden_layer_size, 1], stddev=1), name='w2')
            variable_summaries(w2)
        with tf.name_scope('biases1'):
            b1 = tf.Variable(tf.random_normal([1]), name='b1')
            variable_summaries(b1)
        with tf.name_scope('biases2'):
            b2 = tf.Variable(tf.random_normal([1]), name='b2')
        a = tf.nn.tanh(tf.matmul(X, w1) + b1)
        y_ = tf.nn.tanh(tf.matmul(a, w2) + b2, name="y_predict")

    return y_, w1, w2


def train_nn(x_train, y_train):
    y_, w1, w2 = nn()

    with tf.Session() as sess:
        # 交叉熵
        # loss = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
        # 含正则化的 MSE
        loss = tf.reduce_mean(tf.square(Y - y_)) + \
               tf.contrib.layers.l2_regularizer(w1_regularizer_rate)(w1) + \
               tf.contrib.layers.l2_regularizer(w2_regularizer_rate)(w2)
        tf.summary.scalar('loss', loss)
        # 不含正则化的 MSE
        # loss = tf.reduce_mean(tf.square(Y - y_))
        # 梯度下降
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(TensorBoarddir, sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)

        for steps in range(train_step):
            seq = steps % 19
            # xbatch = x_total[seq:seq + batch_size]
            # xbatch = np.expand_dims(xbatch, axis=1).transpose()
            xbatch = x_train
            # ybatch = y[seq:seq + batch_size]
            ybatch = y_train
            summary, _, cost_ = sess.run([merged, train_op, loss], feed_dict={X: xbatch, Y: ybatch})
            if steps % 100 == 0:
                train_writer.add_summary(summary, steps)
                # print("训练步数: ", steps, " MSE = ", cost_)
                print("训练步数: ", steps, " RMSE = ", pow(cost_, 0.5))
                # print("训练步数: ", steps, " cross entropy = ", cost_)

        saver = tf.train.Saver()
        saver.save(sess, Modeldir)


def prediction(x_test):
    y_, _, _ = nn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # saver.restore(sess, Modeldir)
        saver.restore(sess, Modeldir)
        predict = sess.run(y_, feed_dict={X: x_test})
    return predict


if __name__ == '__main__':
    xtr, xte, ytr = get_data()
    # xtrain_non, xtest_non = get_data_nonnormailized()
    # print(ytr)
    # train_nn(xtr, ytr)
    # print(ytr)
    # print(xte)
    # ypre = prediction(xte)
    # print(ypre)
    # print(xtest_non)
    # ypre1 = prediction(xtrain_non)
    # ypre2 = prediction(xtest_non)
