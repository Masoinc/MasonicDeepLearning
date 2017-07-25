import tensorflow as tf
from numpy.random import RandomState

from Utility.XlsReader import readxlsbycol

batch_size = 8
# 每次迭代的数据量
w1 = tf.Variable(tf.random_normal([1, 41], stddev=1, seed=1))
# 1x41矩阵,stddev(标准差)=1
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 3x1矩阵

DIRECTORY = r"E:\PyCharmProjects\MasonicDeepLearning\DataSet\Bailuyuan.xlsx"
SHEET = "Sheet1"
x = 0
y = 1

x = tf.placeholder(readxlsbycol(DIRECTORY, SHEET, x))
# shape=矩阵维度,None=不定
y_ = tf.placeholder(readxlsbycol(DIRECTORY, SHEET, y))

a = tf.matmul(x, w1)
# x=41x1
# w=1x41
# 矩阵相乘
y = tf.matmul(a, w2)

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# 交叉熵(损失函数)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# 反向传播算法


rdm = RandomState(1)
X = rdm.rand(128, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # 初始化变量
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    print
    "w1:", sess.run(w1)
    print
    "w2:", sess.run(w2)
    print
    "\n"

    # 训练模型
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % 128
        end = (i * batch_size) % 128 + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    # 输出训练后的参数取值。
    print
    "\n"
    print
    "w1:", sess.run(w1)
    print
    "w2:", sess.run(w2)
