import tensorflow as tf
import numpy as np

from Utility.XlsReader import readxlsbycol, getdivideddataset, readxls


def getrandomrow(col, amount):
    region = np.random.rand(0, np.floor(len(col)))


def linerRegression(train_x, train_y, epoch=1000000, rate=0.000001):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    n = train_x.shape[0]
    x = tf.placeholder("float")
    y = tf.placeholder("float")
    w = tf.Variable(tf.random_normal([1]))  # 生成随机权重
    b = tf.Variable(tf.random_normal([1]))

    pred = tf.add(tf.multiply(x, w), b)
    loss = tf.reduce_sum(tf.pow(pred - y, 2))
    optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    print
    'w  start is ', sess.run(w)
    print
    'b start is ', sess.run(b)
    for index in range(epoch):
        # for tx,ty in zip(train_x,train_y):
        # sess.run(optimizer,{x:tx,y:ty})
        sess.run(optimizer, {x: train_x, y: train_y})
        # print 'w is ',sess.run(w)
        # print 'b is ',sess.run(b)
        # print 'pred is ',sess.run(pred,{x:train_x})
        # print 'loss is ',sess.run(loss,{x:train_x,y:train_y})
        # print '------------------'
    print
    'loss is ', sess.run(loss, {x: train_x, y: train_y})
    w = sess.run(w)
    b = sess.run(b)
    return w, b


def predictionTest(test_x, test_y, w, b):
    W = tf.placeholder(tf.float32)
    B = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    pred = tf.add(tf.multiply(X, W), B)
    loss = tf.reduce_mean(tf.pow(pred - Y, 2))
    sess = tf.Session()
    loss = sess.run(loss, {X: test_x, Y: test_y, W: w, B: b})
    return loss


if __name__ == "__main__":
    DIRECTORY = r"E:\PyCharmProjects\MasonicDeepLearning\DataSet\Bailuyuan.xlsx"
    SHEET = "Sheet1"

    xtrain, xtest, ytrain, ytest = getdivideddataset(readxls(DIRECTORY, SHEET), 0.7)

    w, b = linerRegression(xtrain, ytrain)

    print('weights', w)

    print('bias', b)

    loss = predictionTest(xtest, ytest, w, b)
    print("MSE=", loss)
