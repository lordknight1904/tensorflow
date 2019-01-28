import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as cPickle
import numpy as np
import gzip
import time
import json

with open('./config.json') as json_data_file:
    cfg = json.load(json_data_file)


class Interface:
    acc = []
    epoch_set = []
    count = 0

    def __init__(self, model=None, train=(), validation=()):
        self.model = model
        self.train = train
        self.validation = validation
        pass

    def start(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for epoch in range(self.model.cfg['training_epochs']):
                # train the model
                count = 0
                self.model.feed_data(sess, self.train[0], self.train[1])
                while True:
                    try:
                        # data, label = sess.run(next)
                        data, label = self.model.get_next(sess)
                        self.model.train(sess, data, label)
                        count += 1
                        print("\rTraining epoch: {}({}/{})".format(epoch + 1, count,
                                                                   int(len(self.train[0]) / cfg['batch_size'])),
                              end="")
                    except tf.errors.OutOfRangeError:
                        break
                print("")

                # testing
                avg = []
                self.model.feed_data(sess, self.validation[0], self.validation[1])
                while True:
                    try:
                        data, label = self.model.get_next(sess)
                        a = self.model.test(sess, data, label)
                        avg.append(a)
                        print("\rAccuracy: {}".format(np.mean(avg)), end="")
                    except tf.errors.OutOfRangeError:
                        break
                print("")
                self.epoch_set.append(epoch + 1)
                self.acc.append(np.mean(avg))
            plt.plot(self.epoch_set, self.acc, label='Summary')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.show()
