import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


class RNN:
    """
        input_layer, output_layer: {"dtype": type, "size": number}
        cfg: {learning_rate, batch_size}
    """

    def __init__(self, cfg):
        # class variables
        self.cfg = cfg
        # dataset
        self.input_stream = tf.placeholder(tf.float32, [None, self.cfg['n_step'], self.cfg['n_feature']])
        self.label_stream = tf.placeholder(tf.float32, [None, self.cfg['n_feature']])
        data_set = tf.data.Dataset.from_tensor_slices((self.input_stream, self.label_stream)).batch(
            self.cfg['batch_size'])
        self.iterator = data_set.make_initializable_iterator()
        self.next = self.iterator.get_next()

        # building graph
        self.input = tf.placeholder(tf.float32, [None, self.cfg['n_step'], self.cfg['n_feature']])
        self.target = tf.placeholder(tf.float32, [None, self.cfg['n_feature']])
        self.learning_rate = tf.placeholder(tf.float32, None)
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        lstm_cell = tf.contrib.rnn.LSTMCell(self.cfg['n_units'], state_is_tuple=True)
        self.lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        outputs, self.states = tf.nn.dynamic_rnn(self.lstm_cell, tf.identity(self.input), dtype="float32")
        self.outputs = tf.transpose(outputs, [1, 0, 2])
        last = tf.gather(self.outputs, int(self.outputs.get_shape()[0]) - 1, name="last_lstm_output")

        ws = tf.Variable(tf.truncated_normal([self.cfg['n_units'], self.cfg['n_feature']]), name="w")
        bias = tf.Variable(tf.constant(0.1, shape=[self.cfg['n_feature']]), name="b")
        self.prediction = tf.matmul(last, ws) + bias

        # weight = tf.Variable(tf.truncated_normal([self.cfg['n_units'], self.cfg['n_step']]))
        # bias = tf.Variable(tf.constant(0.1, shape=[self.cfg['n_step']]))
        # self.prediction = tf.matmul(last, weight) + bias

        # loss and optimizer
        self.loss = tf.reduce_mean(tf.square(self.prediction - self.target))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg['learning_rate']).minimize(self.loss)

        self.loss_test = tf.reduce_mean(tf.square(self.prediction - self.target), name="loss_mse_test")

        self.loss_sum = tf.summary.scalar("loss_mse_train", self.loss)
        self.loss_test_sum = tf.summary.scalar("loss_mse_test", self.loss_test)
        self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()
        self.merged_sum = tf.summary.merge_all()

    def get_next(self, sess):
        return sess.run(self.next)

    def predict(self, sess, data, label):
        return sess.run(self.prediction, feed_dict={self.input: data, self.target: label})

    def train(self, sess, data, label, current_lr):
        return sess.run([self.loss, self.optimizer, self.merged_sum],
                        feed_dict={self.input: data, self.target: label, self.learning_rate: current_lr,
                                   self.keep_prob: self.cfg['keep_prob']})

    def write(self, writer, train_merged_sum, global_step):
        writer.add_summary(train_merged_sum, global_step=global_step)

    def feed_data(self, sess, data, label):
        sess.run(self.iterator.initializer, feed_dict={self.input_stream: data, self.label_stream: label})

    def test(self, sess, data, label, learning_rate):
        return sess.run([self.loss_test, self.prediction],
                        feed_dict={self.input: data, self.target: label, self.learning_rate: learning_rate,
                                   self.keep_prob: 1})


def dry_run(self, train, label):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        self.feed_data(sess, train, label)

        x, y = self.get_next(sess)
        print(x)
        print(y)
