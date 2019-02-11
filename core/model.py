import tensorflow as tf
import numpy as np


class Model:
    """
        input_layer, output_layer: {"dtype": type, "size": number}
        cfg: {learning_rate, batch_size}
    """

    def __init__(self, cfg,
                 input_layer=None,
                 output_layer=None
                 ):
        if input_layer is None:
            input_layer = {"dtype": np.float64, "size": 0},
        if output_layer is None:
            output_layer = {"dtype": np.float64, "size": 0}
        # class variables
        self.cfg = cfg
        self.input_layer = input_layer
        self.output_layer = output_layer

        # dataset
        self.input_stream = tf.placeholder(self.input_layer['dtype'], [None, self.input_layer['size']])
        self.label_stream = tf.placeholder(self.output_layer['dtype'], [None, self.output_layer['size']])
        training_data_set = tf.data.Dataset.from_tensor_slices((self.input_stream, self.label_stream)).batch(
            self.cfg['batch_size'])
        self.iterator = training_data_set.make_initializable_iterator()
        self.next = self.iterator.get_next()

        # building graph
        self.input = tf.placeholder(tf.float32, [None, self.input_layer['size']], name='input')
        self.prediction = tf.placeholder(tf.float32, [None, self.output_layer['size']], name='prediction')
        # trainable weights and biases
        W = tf.Variable(tf.random_normal([784, 10]), name='w')
        b = tf.Variable(tf.random_normal([10]), name='b')
        activation = tf.nn.softmax(tf.matmul(self.input, W) + b)
        # loss and train algorithm
        cross_entropy = self.prediction * tf.log(activation)
        cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices=1))
        self.optimizer = tf.train.GradientDescentOptimizer(self.cfg['learning_rate']).minimize(cost)

        # model performance
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(activation, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def get_next(self, sess):
        return sess.run(self.next)

    def train(self, sess, data, label):
        sess.run(self.optimizer, feed_dict={self.input: data, self.prediction: label})

    def feed_data(self, sess, data, label):
        sess.run(self.iterator.initializer, feed_dict={self.input_stream: data, self.label_stream: label})

    def test(self, sess, data, label):
        return sess.run(self.accuracy, feed_dict={self.input: data, self.prediction: label})
