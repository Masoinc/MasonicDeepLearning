import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Utility.Normalize import mnormalize
from Utility.XlsReader import readxlsbycol

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Rootdir = os.path.abspath(os.path.dirname(os.getcwd()))
Modeldir = Rootdir + r"\Models\MNLR\MNLR.ckpt"
Datadir = Rootdir + r"\DataSet\HeatPrediction.xlsx"
TensorBoarddir = Rootdir + r"\TensorBoard\MNLR"

Data_Sheet = "Sheet1"

# 训练参数设定
with tf.name_scope('MNLR_Hyper_Parameter'):
    train_step = 15000
    tf.summary.scalar('train_step', train_step)
    regularizer_enabled = True
    w1_regularizer_rate = 0.001
    tf.summary.scalar('w1_regularizer_rate', w1_regularizer_rate)
    w2_regularizer_rate = 0.001
    tf.summary.scalar('w2_regularizer_rate', w2_regularizer_rate)
    hidden_layer_size = 20
    tf.summary.scalar('hidden_layer_size', hidden_layer_size)
    v_amount = 7
    tf.summary.scalar('variable_amount', v_amount)
    early_stopping_rate = 0.0001
    # 取0为不使用early stopping
    tf.summary.scalar('early_stopping_rate', early_stopping_rate)
    # learning_rate = 0.0001
    # 设定衰减学习率以加速学习
    global_step = tf.Variable(0, name="global_step")
    learning_rate = \
        tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.9,
                                   staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    X = tf.placeholder(tf.float32, shape=(None, v_amount), name='x_train')
    # 输入X 19*7(None * 7)
    tf.summary.histogram(name="X", values=X)

    Y = tf.placeholder(tf.float32, shape=[None, 1], name='y_train')
    # 输出Y 19*1(None * 1)
    tf.summary.histogram(name="Y", values=Y)


# 标准化数据，映射至[0,1]
def get_data(non=True):
    with tf.name_scope('MNLR_Data'):
        x_train = []
        x_test = []
        for i in range(1, 8):
            x = readxlsbycol(Datadir, Data_Sheet, i)[1:25]
            x = mnormalize(x)
            x_train.append(x[:19])
            x_test.append(x[19:])
        with tf.name_scope('train_data'):
            x_train = np.array(x_train).T
            y_train = readxlsbycol(Datadir, Data_Sheet, 8)[1:20]
            if non:
                y_train = mnormalize(y_train)
            y_train = np.expand_dims(y_train, axis=1)
            tf.summary.histogram(name="x_train", values=x_train)
            tf.summary.histogram(name="y_train", values=y_train)
        with tf.name_scope('test_data'):
            x_test = np.array(x_test).T
            tf.summary.histogram(name="x_test", values=x_test)
        return x_train, x_test, y_train


def get_data_nonnormailized():
    x_train = []
    x_test = []
    for i in range(1, 8):
        x = readxlsbycol(Datadir, Data_Sheet, i)[1:25]
        x_train.append(x[:19])
        x_test.append(x[19:])

    x_train = np.array(x_train).T  # 列表转矩阵(7*19)
    x_test = np.array(x_test).T

    return x_train, x_test


def nn():
    with tf.name_scope('MNLR_Neural_Network_Layer'):
        with tf.name_scope('weights'):
            w1 = tf.Variable((tf.random_normal([v_amount, hidden_layer_size], stddev=1)), name='weights1')
            tf.summary.histogram(name="weights1", values=w1)
            w2 = tf.Variable(tf.random_normal([hidden_layer_size, 1], stddev=1), name='weights2')
            tf.summary.histogram(name="weights2", values=w2)
        with tf.name_scope('biases'):
            b1 = tf.Variable(tf.random_normal([1]), name='biases1')
            tf.summary.histogram(name="b1", values=b1)
            b2 = tf.Variable(tf.random_normal([1]), name='biases2')
            tf.summary.histogram(name="b2", values=b2)

        with tf.name_scope('predict'):
            # relu激活函数
            # a = tf.nn.relu(tf.matmul(X, w1) + b1)
            # relu训练过程中神经元易"死亡"，出现预测结果为0的情况
            a = tf.nn.tanh(tf.matmul(X, w1) + b1)
            tf.summary.histogram(name="a", values=a)
            y_ = tf.nn.tanh(tf.matmul(a, w2) + b2, name="y_predict")
            tf.summary.histogram(name="y_predict", values=y_)
    return y_, w1, w2


def train_nn(x_train, y_train):
    y_, w1, w2 = nn()

    # 交叉熵
    # loss = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))

    # MSE
    with tf.name_scope('MNLR_Accuracy'):
        if regularizer_enabled:
            loss = tf.reduce_mean(tf.square(Y - y_)) + \
                   tf.contrib.layers.l1_regularizer(w1_regularizer_rate)(w1) + \
                   tf.contrib.layers.l1_regularizer(w2_regularizer_rate)(w2)
            tf.summary.scalar('loss', loss)
        else:
            loss = tf.reduce_mean(tf.square(Y - y_))
            tf.summary.scalar('loss', loss)

    # 反向传播
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(TensorBoarddir, sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)
        cost_prev = 0
        for steps in range(train_step):

            summary, _, cost_ = sess.run([merged, train_op, loss], feed_dict={X: x_train, Y: y_train})
            # Early stopping
            # 减少过拟合
            if steps % 10 == 0 and cost_prev != -1:
                delta = 1 if cost_prev == 0 else (abs(cost_ - cost_prev) / cost_prev)
                cost_prev = cost_
                if delta < early_stopping_rate:
                    print("训练步数: ", steps, " MSE = ", cost_)
                    print("训练步数: ", steps, " RMSE = ", pow(cost_, 0.5))
                    saver = tf.train.Saver()
                    saver.save(sess, Modeldir)
                    print("模型已保存")
                    cost_prev = -1
            if steps % 100 == 0:
                train_writer.add_summary(summary, steps)
                # print("训练步数: ", steps, " MSE = ", cost_)
                print("训练步数: ", steps, " RMSE = ", pow(cost_, 0.5))
                # print("训练步数: ", steps, " cross entropy = ", cost_)

        saver = tf.train.Saver()
        saver.save(sess, Modeldir)


def prediction(x):
    y_, _, _ = nn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # saver.restore(sess, Modeldir)
        saver.restore(sess, Modeldir)
        predict = sess.run(y_, feed_dict={X: x})

    return predict


def prediction_non(x):
    y_ = prediction(x)
    y_non = []

    _, _, y_train = get_data(False)
    for num in y_:
        y_non.append(num * np.max(y_train))
    y_non = np.transpose(y_non)[0]
    return y_non


def prediction_plot(ytr, ypre):
    plt.figure()
    plt.plot(list(range(len(ytr))), ytr, color='b')
    plt.plot(list(range(len(ypre))), ypre, color='r')
    plt.show()


if __name__ == '__main__':
    xtr, xte, ytr = get_data()
    xtrain_non, xtest_non = get_data_nonnormailized()
    # 训练神经网络
    # train_nn(xtr, ytr)

    # 生成测试数据集拟合图
    prediction_plot(ytr, prediction(xtr))

    # 输出预测结果
    # print(prediction(xte))
    # print(prediction_non(xte))
