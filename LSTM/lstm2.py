import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# MasonicProject
# 2017-7-8-0008
# 使用LSTM模型预测人民的名义收视率
# 及获取RMSE等拟合优度参数


# 加载数据
from Utility.ModelAccuracy import getrsqured, getsumse
from Utility.Normalize import mnormalize, unmnormalize
from Utility.XlsReader import readxlsbycol

Rootdir = os.path.abspath(os.path.dirname(os.getcwd()))
Modeldir = Rootdir + r"\Models\LSTM\LSTM.model"
Datadir = Rootdir + "\DataSet\Renmindemingyi.xlsx"
TensorBoarddir = Rootdir + r"\TensorBoard\LSTM\RNN_LSTM"
Data_Sheet = "Sheet1"

# 训练参数设定
with tf.name_scope('LSTM_Hyper_Parameter'):
    # learning_rate = 0.0001
    # 设定衰减学习率以加速学习
    train_step = 2500
    tf.summary.scalar('train_step', train_step)
    global_step = tf.Variable(0, name="global_step")
    learning_rate = \
        tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.9,
                                   staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    regularizer_enabled = False
    w_regularizer_rate = 0.01
    tf.summary.scalar('w1_regularizer_rate', w_regularizer_rate)
    hidden_layer_size = 15
    tf.summary.scalar('hidden_layer_size', hidden_layer_size)
    # 序列参考量
    seq_size = 10
    tf.summary.scalar('seq_size', seq_size)


# 循环神经网络
def rnn():
    with tf.name_scope('LSTM_Neural_Network_Layer'):
        with tf.name_scope('weights'):
            w1 = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W')
            tf.summary.histogram(name="weights1", values=w1)
            # cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
            cell = tf.nn.rnn_cell.BasicRNNCell(hidden_layer_size)
            outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
            w2 = tf.tile(tf.expand_dims(w1, 0), [tf.shape(X)[0], 1, 1])
            tf.summary.histogram(name="weights2", values=w2)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.random_normal([1]), name='b')
            tf.summary.histogram(name="b", values=b)
        with tf.name_scope('predict'):
            y_ = tf.nn.tanh(tf.matmul(outputs, w2) + b)
            y_ = tf.squeeze(y_)
            tf.summary.histogram(name="y_predict", values=y_)
    return y_, w1, w2


# 数据标准化处理
with tf.name_scope('LSTM_Data'):
    data_full = readxlsbycol(Datadir, Data_Sheet, 0)

    data = data_full[:-3]

    data_nomarlized = mnormalize(data)
    data_full_normalized = mnormalize(data_full)

    x_train, y_train, x_train_non = [], [], []
    for i in range(len(data_nomarlized) - seq_size - 1):
        x_train_non.append(np.expand_dims(data[i: i + seq_size], axis=1).tolist())
        x_train.append(np.expand_dims(data_nomarlized[i: i + seq_size], axis=1).tolist())
        y_train.append(data_nomarlized[i + 1: i + seq_size + 1].tolist())
    tf.summary.histogram(name="x_train", values=x_train)
    tf.summary.histogram(name="y_train", values=y_train)

    X = tf.placeholder(tf.float32, [None, seq_size, 1])
    Y = tf.placeholder(tf.float32, [None, seq_size])


# 训练模型
def train_rnn():
    y_, w1, w2 = rnn()

    # 损失函数(MSE)
    with tf.name_scope('LSTM_Accuracy'):
        if regularizer_enabled:
            loss = tf.reduce_mean(tf.square(Y - y_)) + \
                   tf.contrib.layers.l2_regularizer(w_regularizer_rate)(w1) + \
                   tf.contrib.layers.l2_regularizer(w_regularizer_rate)(w2)
            tf.summary.scalar('loss', loss)
        else:
            loss = tf.reduce_mean(tf.square(y_ - Y))
            tf.summary.scalar('loss', loss)

    # 反向传播算法
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    saver = tf.train.Saver(tf.global_variables())

    merged = tf.summary.merge_all()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(TensorBoarddir, sess.graph)

        for step in range(train_step):
            summary, _, loss_ = sess.run([merged, train_op, loss], feed_dict={X: x_train, Y: y_train})
            if step % 100 == 0:
                train_writer.add_summary(summary, step)
                print(step, loss_)
        print("模型已另存至 ", saver.save(sess, Modeldir))
        print("可视化数据已另存至 ", TensorBoarddir)


# 预测
def prediction(data):
    y_, _, _ = rnn()

    saver = tf.train.Saver(tf.global_variables())

    with tf.name_scope('LSTM_Accuracy'):

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(TensorBoarddir, sess.graph)
            saver.restore(sess, Modeldir)

            prev_seq = data[-1]
            predict = []
            for i in range(3):
                next_seq = sess.run(y_, feed_dict={X: [prev_seq]})
                predict.append(next_seq[-1])
                prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

            real = data_full_normalized[-3:]
            print("真实值", real)
            print("预测值", predict)

            MSE = getsumse(predict, real) / 3
            tf.summary.histogram(name="MSE", values=MSE)

            RMSE = pow(MSE, 0.5)
            tf.summary.histogram(name="RMSE", values=RMSE)

            # 拟合优度
            print("MSE = ", MSE, "RMSE = ", RMSE)
            plt.figure()
            plt.plot(list(range(len(data_nomarlized), len(data_nomarlized) + len(predict))), predict, color='b')
            plt.plot(list(range(len(data_full_normalized))), data_full_normalized, color='r')
            plt.show()


# 预测(反归一化)
def prediction_non(data_n):
    y_, _, _ = rnn()

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        saver.restore(sess, Modeldir)

        prev_seq = data_n[-1]
        predict = []
        for i in range(3):
            next_seq = sess.run(y_, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        print(predict)
        predict_unnormalized = []
        for num in predict:
            predict_unnormalized.append(num * np.max(data_full))
        print(predict_unnormalized)
        real = data_full[-3:]
        print(real)
        MSE = getsumse(predict_unnormalized, real) / 3
        RMSE = pow(MSE, 0.5)
        # 拟合优度
        print("MSE = ", MSE, "RMSE = ", RMSE)
        # MSE = getsumse(predict, real) / 3
        # RMSE = pow(MSE, 0.5)
        # 拟合优度
        # print("MSE = ", MSE, "RMSE = ", RMSE)
        plt.figure()
        plt.plot(list(range(len(data), len(data) + len(predict_unnormalized))), predict_unnormalized,
                 color='b')
        plt.plot(list(range(len(data_full))), data_full, color='r')
        plt.show()


if __name__ == '__main__':
    train_rnn()
    # prediction(x_train)
    # prediction_non(x_train)
