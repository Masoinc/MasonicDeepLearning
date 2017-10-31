import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from Utility.ModelAccuracy import getrsqured, getsumse
from Utility.Normalize import mnormalize, unmnormalize
from Utility.XlsReader import readxlsbycol

Rootdir = os.path.abspath(os.path.dirname(os.getcwd()))
Modeldir = Rootdir + r"\Models\LSTM\LSTM.model"
Datadir = Rootdir + "\DataSet\Renmindemingyi.xlsx"
TFRdir = Rootdir + "\DataSet\TFRecords"
TensorBoarddir = Rootdir + r"\TensorBoard\LSTM"
Data_Sheet = "Sheet1"

seq_size = 10

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float64_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


with tf.name_scope('LSTM_Data'):
    data_full = readxlsbycol(Datadir, Data_Sheet, 0)

    data = data_full[:-3]

    data_nomarlized = mnormalize(data)
    data_full_normalized = mnormalize(data_full)

    x_train, y_train, x_train_non = [], [], []
    x_train_non_lowd, x_train_lowd = [], []
    # 另设升维前的X序列
    for i in range(len(data_nomarlized) - seq_size - 1):
        x_train_non.append(np.expand_dims(data[i: i + seq_size], axis=1).tolist())
        # 未标准化的原始数据
        x_train_non_lowd.append(data[i: i + seq_size])
        # 此处对X进行升维操作，新建变量取得升维前数据
        x_train.append(np.expand_dims(data_nomarlized[i: i + seq_size], axis=1).tolist())
        # 标准化后的数据
        x_train_lowd.append(data_nomarlized[i: i + seq_size])
        y_train.append(data_nomarlized[i + 1: i + seq_size + 1].tolist())
        # Y不需要升维
    tf.summary.histogram(name="x_train", values=x_train)
    tf.summary.histogram(name="y_train", values=y_train)

    X = tf.placeholder(tf.float32, [None, seq_size, 1])
    Y = tf.placeholder(tf.float32, [None, seq_size])


# 1维数据写入
def writer(data_set, name):
    # 转换为tfrecord支持的数据
    num = data_set.__len__()

    file = os.path.join(TFRdir, name + '.tfrecords')
    print(file, '正在写入...')
    writer = tf.python_io.TFRecordWriter(file)

    for i in range(num):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'pre10': _float64_feature(data_set[i])
            })
        )
        writer.write(example.SerializeToString())
    writer.close()


# 2维数据写入
def writer2d(data_set, name):
    # 转换为tfrecord支持的数据
    num = data_set.__len__()

    # TODO: 此处len不应该为全部长度，针对时间序列类预测问题，len应为__len__ - seq_size + 1

    file = os.path.join(TFRdir, name + '.tfrecords')
    print(file, '正在写入...')
    writer = tf.python_io.TFRecordWriter(file)

    for i in range(num):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'pre10': tf.train.Feature(float_list=tf.train.FloatList(value=data_set[i]))
                # 'pre10': _float64_feature(data_set[i])
            })
        )
        writer.write(example.SerializeToString())
    writer.close()


def TFR2dreader(filename):
    filename = TFRdir + "\\" + filename + '.tfrecords'
    q = tf.train.string_input_producer([filename], num_epochs=None)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(q)

    batch = tf.train.batch([serialized_example], batch_size=data.__len__())
    features = {
        # 'pre10': tf.VarLenFeature(tf.float32),

        'pre10': tf.FixedLenSequenceFeature(shape=[data.__len__()], dtype=tf.float32, allow_missing=True),
    }
    p = tf.parse_example(batch, features=features)
    # p是一个dict(hashmap)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # 可取出数据的方法
        for i in p:
            # p此时是一个dict,形如{'pre10':SparseTensor}
            # 对p(dict)进行迭代，取出其中的SparseTensor
            return tf.sparse_tensor_to_dense(p[i]).eval()
            # sparse_tensor_to_dense可将SparseTensor格式转为矩阵
            # p = tf.sparse_tensor_to_dense(p.values)

            # num = reader.num_records_produced()
            # print(num.eval())


def TFR2dreaderv(filename):
    filename = TFRdir + "\\" + filename + '.tfrecords'
    q = tf.train.string_input_producer([filename], num_epochs=None)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(q)

    batch = tf.train.batch([serialized_example], batch_size=data.__len__() - seq_size -1)

    features = {
        # 'pre10': tf.FixedLenSequenceFeature(shape=[data.__len__()], dtype=tf.float32, allow_missing=True),
        'pre10': tf.FixedLenFeature(shape=[10], dtype=tf.float32),
    }

    p = tf.parse_example(batch, features=features)
    # p是一个dict(hashmap)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # 可取出数据的方法
        # print(p)
        for i in p:
            x = p[i].eval()
            x = np.expand_dims(x, axis=2).tolist()
            return x
            # p此时是一个pdict,形如{'pre10':SparseTensor}
            # 对p(dict)进行迭代，取出其中的SparseTensor
            # return tf.sparse_tensor_to_dense(p[i]).eval()
            # sparse_tensor_to_dense可将SparseTensor格式转为矩阵
            # p = tf.sparse_tensor_to_dense(p.values)

            # num = reader.num_records_produced()
            # print(num.eval())


if __name__ == '__main__':
    x = TFR2dreaderv("trX")
    print(x)
    # trX = TFR2dreader("trX")
    #
    # # VarLenFeature会自动填充数据，需要进行裁剪
    # trX = np.expand_dims(trX, axis=2).tolist()
    # print(trX)
    # print(x_train)

    # trX_n = TFR2dreader("trX_non")[0:x_train_non.__len__()]
    # # VarLenFeature会自动填充数据，需要进行裁剪
    # trX_n = np.expand_dims(trX_n, axis=2).tolist()
    # print(trX_n)
    # print(x_train_non)

    # trY = TFR2dreader("trY")[:y_train.__len__()]
    # print(trY.tolist())
    # print(y_train)

    # print(x_train.__len__())

    # writer2d(x_train_lowd, "trX")
    # writer2d(x_train_non_lowd, "trX_non")
    # writer2d(y_train, "trY")
