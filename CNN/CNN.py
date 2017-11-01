import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Utility.Normalize import mnormalize
from Utility.XlsReader import readxlsbycol

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Rootdir = os.path.abspath(os.path.dirname(os.getcwd()))
Modeldir = Rootdir + r"\Models\CNN\CNN.ckpt"
Datadir = Rootdir + r"\DataSet\HeatPredictionv2.xlsx"
TensorBoarddir = Rootdir + r"\TensorBoard\CNN"

Data_Sheet = "Sheet1"

# 训练参数设定
with tf.name_scope('CNN_Hyper_Parameter'):
    train_step = 25000
    tf.summary.scalar('train_step', train_step)
    drop_out = True
    regularizer_enabled = True
    w_regularizer_rate = 0.01
    tf.summary.scalar('w_regularizer_rate', w_regularizer_rate)
    hidden_layer1_size = 20
    tf.summary.scalar('hidden_layer_size', hidden_layer1_size)
    hidden_layer2_size = 10
    tf.summary.scalar('hidden_layer_size', hidden_layer2_size)
    hidden_layer3_size = 5
    tf.summary.scalar('hidden_layer_size', hidden_layer3_size)
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

# 卷积层与池化层提取数据
with tf.variable_scope("CNN_Conv_Layer_Data", reuse=None):
    xsina_full = []
    for i in range(1, 20):
        xsina = readxlsbycol(Datadir, "Sheet2", i)[1:22]
        # 19部电视剧x21天
        xsina = mnormalize(xsina)
        xsina_full.append(xsina)
    xsina_full = np.array(xsina_full).T

    xqiyi_full = []
    for i in range(1, 20):
        xqiyi = readxlsbycol(Datadir, "Sheet3", i)[1:22]
        # 19部电视剧x21天
        xqiyi = mnormalize(xqiyi)
        xqiyi_full.append(xqiyi)
    xqiyi_full = np.array(xqiyi_full).T


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


def nn():
    with tf.name_scope('CNN_Neural_Network_Layer'):
        with tf.name_scope('weights'):
            w1 = tf.Variable((tf.random_normal([v_amount, hidden_layer1_size], stddev=1)), name='weights1')
            tf.summary.histogram(name="weights1", values=w1)
            w2 = tf.Variable(tf.random_normal([hidden_layer1_size, hidden_layer2_size], stddev=1), name='weights2')
            tf.summary.histogram(name="weights2", values=w2)
            w3 = tf.Variable(tf.random_normal([hidden_layer2_size, hidden_layer3_size], stddev=1), name='weights3')
            tf.summary.histogram(name="weights3", values=w3)
            w4 = tf.Variable(tf.random_normal([hidden_layer3_size, 1], stddev=1), name='weights4')
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

            tf.summary.histogram(name="a", values=a)
            tf.summary.histogram(name="b", values=b)
            y_ = tf.matmul(c, w4) + b4
            tf.summary.histogram(name="y_predict", values=y_)
    return y_, w1, w2, w3, w4


def train_nn(x_train, y_train):
    y_, w1, w2, w3, w4 = nn()

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
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        # 卷积层参数处理
        with tf.name_scope('CNN_Convolution_Layer'):

            x_sina_unstack = tf.unstack(xsina_full, axis=1)
            # 19部电视剧x21天
            # 21 -> 14 (8)卷积
            # 14 -> 12 (3)池化
            # 12 -> 7 (6)卷积
            # 7 -> 5 (3)池化
            # 5 -> 3 (3)池化
            w1_sina = tf.get_variable('w1_sina', [8, 1, 1, xsina_full.shape[1]],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            w1_sina_unstack = tf.unstack(w1_sina, axis=3)
            b1_sina = tf.get_variable('b1_sina', [1], initializer=tf.constant_initializer(0.1))

            w2_sina = tf.get_variable('w2_sina', [6, 1, 1, xsina_full.shape[1]],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            w2_sina_unstack = tf.unstack(w2_sina, axis=3)
            b2_sina = tf.get_variable('b2_sina', [1], initializer=tf.constant_initializer(0.1))

            x_qiyi_unstack = tf.unstack(xqiyi_full, axis=1)

            w1_qiyi = tf.get_variable('w1_qiyi', [8, 1, 1, xqiyi_full.shape[1]],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            w1_qiyi_unstack = tf.unstack(w1_qiyi, axis=3)
            b1_qiyi = tf.get_variable('b1_qiyi', [1], initializer=tf.constant_initializer(0.1))

            w2_qiyi = tf.get_variable('w2_qiyi', [6, 1, 1, xqiyi_full.shape[1]],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            w2_qiyi_unstack = tf.unstack(w2_qiyi, axis=3)
            b2_qiyi = tf.get_variable('b2_qiyi', [1], initializer=tf.constant_initializer(0.1))

        train_writer = tf.summary.FileWriter(TensorBoarddir, sess.graph)
        # 全局变量初始化
        init = tf.global_variables_initializer()
        sess.run(init)
        # 卷积层运算
        convsI_sina = []
        for i in range(xsina_full.shape[1]):
            xconv = tf.expand_dims(x_sina_unstack[i], 1)
            xconv = tf.expand_dims(xconv, 0)

            convI_sina = tf.nn.conv1d(tf.cast(xconv, tf.float32), w1_sina_unstack[i], stride=1, padding="VALID",
                                      data_format="NHWC")
            # 卷积-8
            convI_sina = tf.nn.bias_add(convI_sina, b1_sina)
            convI_sina_actived = tf.nn.relu(convI_sina)
            convI_sina_actived = tf.expand_dims(convI_sina_actived, 2)
            poolI = tf.nn.max_pool(convI_sina_actived, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
            # 池化-3
            convsI_sina.append(poolI)

        convsII_sina = []
        for i in range(len(convsI_sina)):
            convII_sina = tf.squeeze(convsI_sina[i], axis=-1)
            convII_sina = tf.nn.conv1d(convII_sina, w2_sina_unstack[i], stride=1, padding="VALID", data_format="NHWC")
            convII_sina = tf.nn.bias_add(convII_sina, b2_sina)
            convII_sina_actived = tf.nn.relu(convII_sina)
            convII_sina_actived = tf.expand_dims(convII_sina_actived, 2)
            poolII = tf.nn.max_pool(convII_sina_actived, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding="VALID")
            convsII_sina.append(poolII)
        convs_sina = tf.cast(tf.squeeze(convsII_sina), tf.float64)

        convsI_qiyi = []
        for i in range(xqiyi_full.shape[1]):
            xconv = tf.expand_dims(x_qiyi_unstack[i], 1)
            xconv = tf.expand_dims(xconv, 0)

            convI_qiyi = tf.nn.conv1d(tf.cast(xconv, tf.float32), w1_qiyi_unstack[i], stride=1, padding="VALID",
                                      data_format="NHWC")
            convI_qiyi = tf.nn.bias_add(convI_qiyi, b1_qiyi)
            convI_qiyi_actived = tf.nn.relu(convI_qiyi)
            convI_qiyi_actived = tf.expand_dims(convI_qiyi_actived, 2)
            poolI = tf.nn.max_pool(convI_qiyi_actived, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
            convsI_qiyi.append(poolI)

        convsII_qiyi = []
        for i in range(len(convsI_qiyi)):
            convII_qiyi = tf.squeeze(convsI_qiyi[i], axis=-1)
            convII_qiyi = tf.nn.conv1d(convII_qiyi, w2_qiyi_unstack[i], stride=1, padding="VALID", data_format="NHWC")
            convII_qiyi = tf.nn.bias_add(convII_qiyi, b2_qiyi)
            convII_qiyi_actived = tf.nn.relu(convII_qiyi)
            convII_qiyi_actived = tf.expand_dims(convII_qiyi_actived, 2)
            poolII = tf.nn.max_pool(convII_qiyi_actived, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding="VALID")
            convsII_qiyi.append(poolII)
        convs_qiyi = tf.cast(tf.squeeze(convsII_qiyi), tf.float64)

        # 持久化卷积层参数，用于预测
        with sess.as_default():

            np.save("convs_sina.npy", convs_sina.eval())
            np.save("convs_qiyi.npy", convs_qiyi.eval())
            pre_convs_sina, convs_sina = convs_sina.eval()[16:19], convs_sina.eval()[0:16]
            pre_convs_qiyi, convs_qiyi = convs_qiyi.eval()[16:19], convs_qiyi.eval()[0:16]

        x_train = np.hstack((x_train, convs_sina))
        x_train = np.hstack((x_train, convs_qiyi))
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
