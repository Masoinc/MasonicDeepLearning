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
Modeldir = r"E:\PyCharmProjects\MasonicDeepLearning\Models\LSTM_MarkI\LSTM-MarkI_I.model"
Datadir = Rootdir + "\DataSet\LSTM_MarkI\白鹿原.xlsx"
TensorBoarddir = Rootdir + r"\TensorBoard\LSTM\LSTM_MarkI_I"
Data_Sheet = "白鹿原"
# 以白鹿原播放量为数据库构建的1层LSTM模型


# 训练参数设定
with tf.name_scope('LSTM_Hyper_Parameter'):
    # learning_rate = 0.0001
    # 设定衰减学习率以加速学习
    train_step = 1000
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
            w1 = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W')
            mcell = []
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_layer_size)
            # 单层LSTM
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            # drop-out层
            mcell.append(cell)
        outputs, states = tf.nn.dynamic_rnn(cell, inputs=X, dtype=tf.float32, time_major=True)
        # h_state = states[-1][1]
        # TODO: w2?
        w2 = tf.tile(input=tf.expand_dims(w1, 0), multiples=[tf.shape(X)[0], 1, 1])
        # 此处添加偏置项不当可能导致图像整体平移
        # b = tf.Variable(tf.random_normal([1]), name='b')
        y_ = tf.nn.tanh(tf.matmul(outputs, w2))
        y_ = tf.squeeze(y_)

    return y_, w1, w2  # 数据标准化处理


with tf.name_scope('LSTM_Data'):
    qiyi_f = readxlsbycol(Datadir, Data_Sheet, 1)

    q_X = mnormalize(qiyi_f)
    q_seq, y_seq = [], []
    for i in range(len(q_X) - seq_size - 1):
        q_seq.append(np.expand_dims(q_X[i:i + seq_size], axis=1).tolist())
        y_seq.append(q_X[i + 1:i + seq_size + 1].tolist())
    trX = q_seq[:151]
    trY = y_seq[:151]
    teX = q_seq[151]
    teY = q_seq[151:]

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
            summary, _, loss_ = sess.run([merged, train_op, loss], feed_dict={X: trX, Y: trY})
            if step % 100 == 0:
                train_writer.add_summary(summary, step)
                print(step, loss_)
            if step % 500 == 0 and step >= 500:
                print("模型已另存至 ", saver.save(sess, Modeldir))
                print("可视化数据已另存至 ", TensorBoarddir)
        print("模型已另存至 ", saver.save(sess, Modeldir))
        print("可视化数据已另存至 ", TensorBoarddir)


# 预测
def prediction():
    y_, _, _ = rnn()

    preseq = 30

    saver = tf.train.Saver(tf.global_variables())

    with tf.name_scope('LSTM_Accuracy'):
        with tf.Session() as sess:
            saver.restore(sess, Modeldir)

            prev_seq = teX
            predict = []
            for i in range(q_X.__len__() - 150):
                next_seq = sess.run(y_, feed_dict={X: [prev_seq]})
                predict.append(next_seq[-1])
                prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

            real = q_X[151:]
            print("真实值", real)
            print("预测值", predict)

            MSE = getsumse(predict, real) / (q_X.__len__()-150)
            tf.summary.histogram(name="MSE", values=MSE)

            RMSE = pow(MSE, 0.5)
            tf.summary.histogram(name="RMSE", values=RMSE)

            # 拟合优度
            print("MSE = ", MSE, "RMSE = ", RMSE)
            # plt.figure()
            # plt.plot(list(range(len(data_nomarlized), len(data_nomarlized) + len(predict))), predict, color='b')
            # plt.plot(list(range(len(data_full_normalized))), data_full_normalized, color='r')
            # plt.show()


if __name__ == '__main__':
    # train_rnn()
    prediction()
