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

Rootdir = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
# Modeldir = Rootdir + r"\Models\LSTM\LSTM-I"
Modeldir = r"E:\PyCharmProjects\MasonicDeepLearning\Models\LSTM_MarkII\LSTM-MarkII_II.model"
Datadir = Rootdir + "\DataSet\LSTM_MarkII\白鹿原_MIX.xlsx"
TensorBoarddir = Rootdir + r"\TensorBoard\LSTM\LSTM_MarkII"
Data_Sheet = "白鹿原_MIX"
# 以白鹿原播放量和微博热度为数据库构建的双变量单层LSTM模型


# 训练参数设定
with tf.name_scope('LSTM_Hyper_Parameter'):
    # learning_rate = 0.0001
    # 设定衰减学习率以加速学习
    train_step = 10000
    global_step = tf.Variable(0, name="global_step")
    learning_rate = \
        tf.train.exponential_decay(learning_rate=0.01, global_step=global_step, decay_steps=100, decay_rate=0.9,
                                   staircase=True)
    regularizer_enabled = True
    w_regularizer_rate = 0.5
    hidden_layer_size = 30
    # 序列参考量
    seq_size = 10
    keep_prob = 0.9


# 循环神经网络
def rnn():
    with tf.name_scope('LSTM_Neural_Network_Layer'):
        with tf.name_scope('weights'):
            w1 = tf.Variable(tf.random_normal([hidden_layer_size, 1]))
            # q_w1(30, 1)
            mcell = []
            for layer in range(2):
                cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_layer_size)
                # 单层LSTM
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
                # drop-out层
                mcell.append(cell)
            mcell = tf.nn.rnn_cell.MultiRNNCell(cells=mcell, state_is_tuple=True)
            #
            outputs, states = tf.nn.dynamic_rnn(mcell, inputs=X, dtype=tf.float32, time_major=True)
            w2 = tf.tile(input=tf.expand_dims(w1, 0), multiples=[tf.shape(X)[0], 1, 1])
            y_ = tf.nn.tanh(tf.matmul(outputs, w2))
            # (?, 10, 30)*(?, 30, 1) = (?, 10, 1)
            # <==> (10, 30)*(30, 1) = (10, 1)

            y_ = tf.squeeze(y_)

    return y_, w1, w2


# # 循环神经网络
# def rnn():
#     with tf.name_scope('LSTM_Neural_Network_Layer'):
#         with tf.name_scope('weights'):
#             q_w1 = tf.Variable(tf.random_normal([hidden_layer_size, 1]))
#             # q_w1(30, 1)
#             cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_layer_size)
#
#             outputs, states = tf.nn.dynamic_rnn(cell, inputs=X1, dtype=tf.float32, time_major=True)
#             # outputs(?, 10, 30)
#             # X1(?, 10, 1)
#             multi = [tf.shape(X1)[0], 1, 1]
#             # multi(?, 1, 1)
#             q_w1_i = tf.expand_dims(q_w1, 0)
#             # (1, 30 ,1)
#             q_w2 = tf.tile(input=q_w1_i, multiples=multi)
#             # (?, 30, 1)
#             # 通过tile方法共享参数
#
#             # 此处添加偏置项不当可能导致图像整体平移
#             # b = tf.Variable(tf.random_normal([1]), name='b')
#             y_ = tf.nn.tanh(tf.matmul(outputs, q_w2))
#             # (?, 10, 30)*(?, 30, 1) = (?, 10, 1)
#             # <==> (10, 30)*(30, 1) = (10, 1)
#
#
#
#             y_ = tf.squeeze(y_)
#
#     return y_, q_w1, q_w2


# 数据标准化处理
with tf.name_scope('LSTM_Data'):
    qiyi_f = readxlsbycol(Datadir, Data_Sheet, 1)
    sina_f = readxlsbycol(Datadir, Data_Sheet, 3)

    q_X = mnormalize(qiyi_f)
    s_X = mnormalize(sina_f)

    q_seq, s_seq, y_seq = [], [], []
    for i in range(len(q_X) - seq_size - 1):
        q_seq.append(np.expand_dims(q_X[i:i + seq_size], axis=1).tolist())
        s_seq.append(np.expand_dims(s_X[i:i + seq_size], axis=1).tolist())
        y_seq.append(q_X[i + 1:i + seq_size + 1].tolist())

    q_trX = q_seq[:151]
    q_teX = q_seq[151:]
    s_trX = s_seq[:151]
    s_teX = s_seq[151:]
    y_tr = y_seq[:151]

    trX = np.concatenate((q_trX, s_trX), axis=2)
    teX = np.concatenate((q_teX, s_teX), axis=2)
    X = tf.placeholder(tf.float32, [None, seq_size, 2])

    # X1 = tf.placeholder(tf.float32, [None, seq_size, 1])
    X2 = tf.placeholder(tf.float32, [None, seq_size, 1])
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
            # summary, _, loss_ = sess.run([merged, train_op, loss], feed_dict={X1: q_trX, X2: s_trX, Y: y_tr})
            summary, _, loss_ = sess.run([merged, train_op, loss], feed_dict={X: trX, Y: y_tr})
            if step % 100 == 0:
                train_writer.add_summary(summary, step)
                print(step, loss_)
            if step % 500 == 0 and step >= 500:
                print("模型已另存至 ", saver.save(sess, Modeldir))
                print("可视化数据已另存至 ", TensorBoarddir)
        print("模型已另存至 ", saver.save(sess, Modeldir))
        print("可视化数据已另存至 ", TensorBoarddir)


# 预测(单值)
def prediction():
    y_, _, _ = rnn()

    saver = tf.train.Saver(tf.global_variables())

    with tf.name_scope('LSTM_Accuracy'):
        with tf.Session() as sess:
            saver.restore(sess, Modeldir)

            predict1 = sess.run(y_, feed_dict={X: teX})
            predict2 = np.array(predict1).T

            real = q_X[151 + 11:]
            # 此处的11是之前 -seq_size -1导致的
            predict3 = predict2[-1]
            print("真实值", real)
            print("预测值", predict3)
            #
            MSE = getsumse(predict3, real) / (q_X.__len__() - 150)
            # tf.summary.histogram(name="MSE", values=MSE)
            #
            RMSE = pow(MSE, 0.5)
            # tf.summary.histogram(name="RMSE", values=RMSE)
            # # 拟合优度
            #
            print("MSE = ", MSE, "RMSE = ", RMSE)


#
# # 预测(递推)
# def predictionx():
#     y_, _, _ = rnn()
#
#     saver = tf.train.Saver(tf.global_variables())
#
#     with tf.name_scope('LSTM_Accuracy'):
#         with tf.Session() as sess:
#             saver.restore(sess, Modeldir)
#
#             predict = sess.run(y_, feed_dict={X: q_teX})
#
#             print("真实值", q_X[:10])
#             print("预测值", predict[:10][-1])
#             #
#             MSE = getsumse(predict[:10][-1], q_X[:10]) / 10
#             # tf.summary.histogram(name="MSE", values=MSE)
#             #
#             RMSE = pow(MSE, 0.5)
#             # tf.summary.histogram(name="RMSE", values=RMSE)
#             #
#             # # 拟合优度
#             print("MSE = ", MSE, "RMSE = ", RMSE)


if __name__ == '__main__':
    # train_rnn()
    prediction()
