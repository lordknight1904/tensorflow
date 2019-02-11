import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


class Model:

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
        # loss and optimizer
        self.loss = tf.reduce_mean(tf.square(self.prediction - self.target))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg['learning_rate']).minimize(self.loss)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()
        self.summaries = tf.summary.merge_all()

    # train the model
    def train(self):
        pass

    # after model train, test the model
    def test(self):
        pass
