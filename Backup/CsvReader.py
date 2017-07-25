import tensorflow as tf

if __name__ == '__main__':

    filename_queue = tf.train.input_producer(["E:\PyCharmProjects\MasonicDeepLearning\DataSet\Bailuyuan.csv"])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    print(key, value)
    # print(tf.string_to_number(key))
    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.

    col1, col2 = tf.decode_csv(value, [[1.0], [1.0]], ",")

    with tf.Session() as sess:
        #     # sess.run(tf.string_to_number(key))
        #     # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        #
        for i in range(50):
            # Retrieve a single instance:
            example, label = sess.run(tf.stack([col1, col2]))
            print(label)
        coord.request_stop()
        coord.join(threads)
