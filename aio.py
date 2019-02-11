import os
import tensorflow as tf
tf.reset_default_graph()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# dataset
input_stream = tf.placeholder(tf.float32, [None, 2], name='data_set_input_stream')
label_stream = tf.placeholder(tf.float32, None)
training_data_set = tf.data.Dataset.from_tensor_slices((input_stream, label_stream)).batch(5)
iterator = training_data_set.make_initializable_iterator()
next_data = iterator.get_next()

x = tf.placeholder(tf.float32, None, name='x')
y = tf.placeholder(tf.float32, None, name='y')

z = tf.add(x, y, name='z')
target = tf.placeholder(tf.float32, None, name='y')
accuracy = tf.metrics.accuracy(target, z)
tf.summary.histogram("accuracy", accuracy)
# tf.summary.histogram(“softmax_w”, softmax_w)

data_x = [(x, x*10) for x in range(100)]
data_y = [x + x*10 for x in range(100)]

if __name__ == '__main__':
    with tf.Session() as sess:
        # init variables
        # tf.reset_default_graph()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        writer = tf.summary.FileWriter(ROOT_DIR + '/logs/train/1', sess.graph)
        # train_writer = tf.summary.FileWriter(ROOT_DIR + '/logs/train/1', sess.graph)

        merge = tf.summary.merge_all()
        # input data feed
        sess.run(iterator.initializer, feed_dict={input_stream: data_x, label_stream: data_y})
        while True:
            try:
                data, data2 = sess.run(next_data)
                acc, _ = sess.run([merge, z], feed_dict={x: data[:, 0], y: data[:, 1], target: data2})
            except tf.errors.OutOfRangeError:
                break
        writer.add_summary(acc, 1)
