import tensorflow as tf
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import tensorflow.tensorboard as tb
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


# 训练参数
with tf.name_scope('Model_Settings'):
    train_step = 10000
    tf.summary.scalar('train_step', train_step)
    w1_regularizer_rate = 0.01
    tf.summary.scalar('w1_regularizer_rate', w1_regularizer_rate)
    w2_regularizer_rate = 0.01
    tf.summary.scalar('w2_regularizer_rate', w2_regularizer_rate)
    hidden_layer_size = 15
    tf.summary.scalar('hidden_layer_size', hidden_layer_size)
    v_amount = 7
    tf.summary.scalar('variable_amount', v_amount)
    # batch_size = 3
    # learning_rate = 0.0001
    # 指数衰减学习率
    global_step = tf.Variable(0, name="global_step")
    learning_rate = \
        tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.9,
                                   staircase=True, name="learning_rate")
    tf.summary.scalar('learning_rate', learning_rate)

# 标准化(无量纲化)数据
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
y_train = readxlsbycol(Datadir, data_sheet, 8)[1:20]
y_train = mnormalize(y_train)
with tf.name_scope('Data'):
    with tf.name_scope('training_data'):
        x_train = np.array(x_train).T  # 列表转矩阵(7*19)
        y_train = np.expand_dims(y_train, axis=1)
        tf.summary.histogram(name="x_train", values=x_train)
        tf.summary.histogram(name="y_train", values=y_train)
    with tf.name_scope('test_data'):
        x_test = np.array(x_test).T
        tf.summary.histogram(name="x_test", values=x_test)
    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, shape=(None, v_amount), name='x_train')
        tf.summary.histogram(name="X", values=X)
        # 输入参数 7*19(None * 7)
    with tf.name_scope('Output'):
        Y = tf.placeholder(tf.float32, shape=[None, 1], name='y_train')
        tf.summary.histogram(name="Y", values=Y)
        # 输出结果 1*19(None * 1)

with tf.name_scope('Nerual_Network_Layer'):
    with tf.name_scope('weights1'):
        w1 = tf.Variable((tf.random_normal([v_amount, hidden_layer_size], stddev=1)), name='w1')
        tf.summary.histogram(name="w1", values=w1)
    with tf.name_scope('weights2'):
        w2 = tf.Variable(tf.random_normal([hidden_layer_size, 1], stddev=1), name='w2')
        tf.summary.histogram(name="w2", values=w2)
    with tf.name_scope('biases1'):
        b1 = tf.Variable(tf.random_normal([1]), name='b1')
        tf.summary.histogram(name="b1", values=b1)
    with tf.name_scope('biases2'):
        b2 = tf.Variable(tf.random_normal([1]), name='b2')
        tf.summary.histogram(name="b2", values=b2)
    a = tf.nn.tanh(tf.matmul(X, w1) + b1)
    with tf.name_scope('y_predict'):
        y_ = tf.nn.tanh(tf.matmul(a, w2) + b2, name="y_predict")
        tf.summary.histogram(name="y_pre", values=y_)

    # 交叉熵
    # loss = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
    # 含正则化的 MSE
    loss = tf.reduce_mean(tf.square(Y - y_)) + \
           tf.contrib.layers.l1_regularizer(w1_regularizer_rate)(w1) + \
           tf.contrib.layers.l1_regularizer(w2_regularizer_rate)(w2)
    tf.summary.scalar('loss', loss)

    # 不含正则化的 MSE
    # loss = tf.reduce_mean(tf.square(Y - y_))
    # 梯度下降
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    merged = tf.summary.merge_all()
    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        writer = tf.summary.FileWriter(TensorBoarddir, sess.graph)
        cost_prev = 0
        for steps in range(train_step):
            seq = steps % 19
            # xbatch = x_total[seq:seq + batch_size]
            # xbatch = np.expand_dims(xbatch, axis=1).transpose()
            xbatch = x_train
            # ybatch = y[seq:seq + batch_size]
            ybatch = y_train
            summary, _, cost_ = sess.run([merged, train_op, loss], feed_dict={X: xbatch, Y: ybatch})

            if steps % 10 == 0 and cost_prev != -1:
                delta = 1 if cost_prev == 0 else (math.pow(math.pow(cost_ - cost_prev, 2), 0.5) / cost_prev)
                cost_prev = cost_
                if delta < 0.0001:
                    print("训练步数: ", steps, " MSE = ", cost_)
                    print("训练步数: ", steps, " RMSE = ", pow(cost_, 0.5))
                    saver = tf.train.Saver()
                    saver.save(sess, Modeldir)
                    print("模型已保存")
                    cost_prev = -1
            if steps % 100 == 0:
                writer.add_summary(summary, steps)
                print("训练步数: ", steps, " MSE = ", cost_)
                print("训练步数: ", steps, " RMSE = ", pow(cost_, 0.5))
                # print("训练步数: ", steps, " cross entropy = ", cost_)
