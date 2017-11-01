import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Utility.Normalize import mnormalize
from Utility.XlsReader import readxlsbycol

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Rootdir = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
Modeldir = Rootdir + r"\Models\CNN\CNN.ckpt"
Datadir = Rootdir + r"\DataSet\HeatPredictionv2.xlsx"
TensorBoarddir = Rootdir + r"\TensorBoard\CNN"

Data_Sheet = "Sheet1"


# 标准化数据，映射至[0,1]
def get_data(non=True):
    with tf.name_scope('CNN_Data'):
        x_train = []
        x_test = []
        for i in range(1, 7):
            x = readxlsbycol(Datadir, Data_Sheet, i)[1:20]
            x = mnormalize(x)
            x_train.append(x[:16])
            x_test.append(x[16:])

        x_train = np.array(x_train).T
        y_train = readxlsbycol(Datadir, Data_Sheet, 7)[1:17]
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
        x = readxlsbycol(Datadir, Data_Sheet, i)[1:24]
        x_train.append(x[:19])
        x_test.append(x[19:])

    x_train = np.array(x_train).T  # 列表转矩阵(7*19)
    x_test = np.array(x_test).T

    return x_train, x_test


# 超参数设定
with tf.name_scope('CNN_Hyper_Parameter'):
    train_step = 25000
    tf.summary.scalar('train_step', train_step)
    drop_out = True
    regularizer_enabled = True
    w_regularizer_rate = 0.01
    tf.summary.scalar('w_regularizer_rate', w_regularizer_rate)
    layer1_size = 20
    tf.summary.scalar('hidden_layer_size', layer1_size)
    layer2_size = 10
    tf.summary.scalar('hidden_layer_size', layer2_size)
    layer3_size = 5
    tf.summary.scalar('hidden_layer_size', layer3_size)
    v_amount = 12
    tf.summary.scalar('variable_amount', v_amount)
    early_stopping_rate = 0
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

with tf.name_scope('CNN_Training_Parameter'):
    # 全连接层参数
    with tf.name_scope('weights'):
        w1 = tf.Variable((tf.random_normal([v_amount, layer1_size])), name='weights1')
        tf.summary.histogram(name="weights1", values=w1)
        w2 = tf.Variable(tf.random_normal([layer1_size, layer2_size]), name='weights2')
        tf.summary.histogram(name="weights2", values=w2)
        w3 = tf.Variable(tf.random_normal([layer2_size, layer3_size]), name='weights3')
        tf.summary.histogram(name="weights3", values=w3)
        w4 = tf.Variable(tf.random_normal([layer3_size, 1]), name='weights4')
        tf.summary.histogram(name="weights4", values=w4)
    with tf.name_scope('biases'):
        b1 = tf.Variable(tf.random_normal([1]), name='biases1')
        tf.summary.histogram(name="b1", values=b1)
        b2 = tf.Variable(tf.random_normal([1]), name='biases2')
        tf.summary.histogram(name="b2", values=b2)
        b3 = tf.Variable(tf.random_normal([1]), name='biases3')
        tf.summary.histogram(name="b3", values=b3)
        b4 = tf.Variable(tf.random_normal([1]), name='biases4')
        tf.summary.histogram(name="b4", values=b4)
    # 卷积层参数
        # convolutional weights
        c_w = {
            # sina
            's_c1' : tf.get_variable('w1_sina', [8, 1, 1, 19]),
            's_c1b' : tf.get_variable('b1_sina', [1]),
            's_c2': tf.get_variable('w2_sina', [6, 1, 1, 19]),
            's_c2b': tf.get_variable('b1_sina', [1]),
        }

def cnn(X, SinaX, QiyiX):

    with tf.name_scope('CNN_Convolution_Layer'):
        c_unstack = tf.unstack(SinaX, axis=1)
        # (21,)
        # 19部电视剧x21天
        # 21 -> 14 (8)卷积
        # 14 -> 12 (3)池化
        # 12 -> 7 (6)卷积
        # 7 -> 5 (3)池化
        # 5 -> 3 (3)池化
        c1 = tf.get_variable('w1_sina', [8, 1, 1, SinaX.shape[1]])
        # (8, 1, 1, 19)
        # (8, 1, 1, 19)
        c1_unstack = tf.unstack(c1, axis=3)
        # 拆分成单个电视剧的数据后进行卷积池化
        # 19x(8, 1, 1)
        c1b = tf.get_variable('b1_sina', [1])
        # (1,)
        c2 = tf.get_variable('w2_sina', [6, 1, 1, SinaX.shape[1]])
        # (6, 1, 1, 19)
        c2_unstack = tf.unstack(c2, axis=3)
        # (6, 1, 1)
        c2b = tf.get_variable('b2_sina', [1])
        # (1,)
        x_qiyi_unstack = tf.unstack(xqiyi_full, axis=1)

        w1_qiyi = tf.get_variable('w1_qiyi', [8, 1, 1, xqiyi_full.shape[1]])
        w1_qiyi_unstack = tf.unstack(w1_qiyi, axis=3)
        b1_qiyi = tf.get_variable('b1_qiyi', [1])

        w2_qiyi = tf.get_variable('w2_qiyi', [6, 1, 1, xqiyi_full.shape[1]])
        w2_qiyi_unstack = tf.unstack(w2_qiyi, axis=3)
        b2_qiyi = tf.get_variable('b2_qiyi', [1])

    # train_writer = tf.summary.FileWriter(TensorBoarddir, sess.graph)
    # 全局变量初始化
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # 卷积层运算

    out_sina = []
    for i in range(SinaX.shape[1]):
        Sina_perX = tf.expand_dims(c_unstack[i], 1)
        # (1, 21, 1)
        Sina_perX = tf.expand_dims(Sina_perX, 0)
        # (1, 21, 1, 1)

        cI = tf.nn.conv1d(tf.cast(Sina_perX, tf.float32), c1_unstack[i], stride=1, padding="VALID",
                          data_format="NHWC")
        # Given an input tensor of shape
        # [batch, in_width, in_channels]
        # 此处为(8, 1, 1)
        # 结果(1, 21, 1, 1)->(1, 14, 1, 1)
        #
        # 卷积-8
        cI = tf.nn.bias_add(cI, c1b)
        cI = tf.nn.relu(cI)
        cI = tf.expand_dims(cI, 2)
        pI = tf.nn.max_pool(cI, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        # (1, 12, 1, 1)
        # 池化-3

        cII = tf.squeeze(pI, axis=-1)
        # (1, 7, 1, 1)
        cII = tf.nn.conv1d(cII, c2_unstack[i], stride=1, padding="VALID", data_format="NHWC")
        cII = tf.nn.bias_add(cII, c2b)
        cII = tf.nn.relu(cII)
        cII = tf.expand_dims(cII, 2)
        pII = tf.nn.max_pool(cII, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding="VALID")
        out_sina.append(pII)
    Sina_out = tf.cast(tf.squeeze(out_sina), tf.float64)

    out_qiyi = []
    for i in range(xqiyi_full.shape[1]):
        Sina_perX = tf.expand_dims(x_qiyi_unstack[i], 1)
        Sina_perX = tf.expand_dims(Sina_perX, 0)

        cI = tf.nn.conv1d(tf.cast(Sina_perX, tf.float32), w1_qiyi_unstack[i], stride=1, padding="VALID",
                          data_format="NHWC")
        cI = tf.nn.bias_add(cI, b1_qiyi)
        cI = tf.nn.relu(cI)
        cI = tf.expand_dims(cI, 2)
        pI = tf.nn.max_pool(cI, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding="VALID")

        cII = tf.squeeze(pI, axis=-1)
        cII = tf.nn.conv1d(cII, w2_qiyi_unstack[i], stride=1, padding="VALID", data_format="NHWC")
        cII = tf.nn.bias_add(cII, b2_qiyi)
        cII = tf.nn.relu(cII)
        cII = tf.expand_dims(cII, 2)
        pII = tf.nn.max_pool(cII, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding="VALID")
        out_qiyi.append(pII)
    Qiyi_out = tf.cast(tf.squeeze(out_qiyi), tf.float64)

    X = np.hstack((X, Sina_out))
    X = np.hstack((X, Qiyi_out))

    with tf.name_scope('predict'):
        # relu激活函数
        # a = tf.nn.relu(tf.matmul(X, w1) + b1)
        # relu训练过程中神经元易"死亡"，出现预测结果为定值的情况
        # a = tf.nn.relu(tf.matmul(X, w1) + b1)
        a = tf.matmul(X, w1) + b1
        # b = tf.nn.relu(tf.matmul(a, w2) + b2)
        b = tf.matmul(a, w2) + b2
        c = tf.nn.tanh(tf.matmul(b, w3) + b3)
        # keep_prob = tf.placeholder(tf.float32)
        if drop_out:
            b = tf.nn.dropout(b, 0.01)

        y_ = tf.matmul(c, w4) + b4

def train_nn(x_train, y_train):
    # 交叉熵
    # loss = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))

    # MSE
    with tf.name_scope('CNN_Accuracy'):
        if regularizer_enabled:
            loss = tf.reduce_mean(tf.square(Y - y_)) + \
                   tf.contrib.layers.l1_regularizer(w_regularizer_rate)(w1) + \
                   tf.contrib.layers.l1_regularizer(w_regularizer_rate)(w2) + \
                   tf.contrib.layers.l1_regularizer(w_regularizer_rate)(w3) + \
                   tf.contrib.layers.l1_regularizer(w_regularizer_rate)(w4)
            tf.summary.scalar('loss', loss)
        else:
            loss = tf.reduce_mean(tf.square(Y - y_))
            tf.summary.scalar('loss', loss)
    # 梯度下降算法
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    merged = tf.summary.merge_all()  # 卷积层参数处理

    # # 持久化卷积层参数，用于预测
    # with sess.as_default():
    #
    #     np.save("convs_sina.npy", Sina_out.eval())
    #     np.save("convs_qiyi.npy", Qiyi_out.eval())
    #     pre_convs_sina, Sina_out = Sina_out.eval()[16:19], Sina_out.eval()[0:16]
    #     pre_convs_qiyi, Qiyi_out = Qiyi_out.eval()[16:19], Qiyi_out.eval()[0:16]

    x_train = np.hstack((x_train, Sina_out))
    x_train = np.hstack((x_train, Qiyi_out))
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
    y_, _, _, _, _ = nn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # saver.restore(sess, Modeldir)
        saver.restore(sess, Modeldir)
        predict = sess.run(y_, feed_dict={X: x})

    return predict


def prediction_plot(ytr, ypre):
    plt.figure()
    plt.plot(list(range(len(ytr))), ytr, color='b')
    plt.plot(list(range(len(ypre))), ypre, color='r')
    plt.show()


if __name__ == '__main__':
    xtr, xte, ytr = get_data()
    # 训练神经网络
    train_nn(xtr, ytr)

    # 读取卷积层参数，获取完整数据集
    # conv_sina = np.load("convs_sina.npy")
    # conv_qiyi = np.load("convs_qiyi.npy")
    # xtr = np.concatenate((xtr, conv_sina[0:16]), axis=1)
    # xtr = np.concatenate((xtr, conv_qiyi[0:16]), axis=1)
    # xte = np.concatenate((xte, conv_sina[16:19]), axis=1)
    # xte = np.concatenate((xte, conv_qiyi[16:19]), axis=1)

    # 生成测试数据集拟合图
    # prediction_plot(ytr, prediction(xtr))

    # 输出预测结果
    # print(prediction(xte))

    # SSE = 0
    # yte = prediction(xtr)
    # print(yte)
    # for i in range(len(ytr)):
    #     SSE += pow(ytr[i]-yte[i], 2)
    # MSE = SSE/len(ytr)
    # print("MSE=",MSE)
    # ytr, prediction(xtr)

    # print(prediction_non(xte))
