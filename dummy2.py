import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import math
import os

with open('./config.json') as json_data_file:
    cfg = json.load(json_data_file)


class Interface:
    acc = []
    epoch_set = []
    count = 0

    """
        cfg: {training_epochs, batch_size}
    """

    def __init__(self, cfg, model=None, train=(), test=()):
        self.cfg = cfg
        self.model = model
        self.train = train
        self.test = test
        self._temperature = []
        self.logs_dir = "logs"
        self.plots_dir = "images"
        self.train_X = train[0]
        self.train_y = train[1]
        self.test_X = test[0]
        self.test_y = test[1]

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        # batch_indices =
        # random.shuffle(batch_indices)
        for j in range(num_batches):
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.cfg["n_step"]}
            yield batch_X, batch_y

    def start(self):
        global_step = 0

        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            writer = tf.summary.FileWriter(os.path.join("./logs", 'ma_modal'))
            writer.add_graph(sess.graph)
            learning_rates_to_use = [
                self.cfg['init_learning_rate'] * (
                        self.cfg['learning_rate_decay'] ** max(float(i + 1 - self.cfg['init_epoch']), 0.0)
                ) for i in range(self.cfg['max_epoch'])]
            for epoch in range(self.cfg['max_epoch']):
                epoch_step = 0
                global_step += 1
                current_lr = learning_rates_to_use[epoch]
                # train the model
                # self.model.feed_data(sess, self.train[0], self.train[1])
                for batch_X, batch_y in self.generate_one_epoch(self.cfg['batch_size']):
                        # data, label = self.model.get_next(sess)
                        train_loss, _, train_merged_sum = self.model.train(sess, batch_X, batch_y, current_lr)
                        self.model.write(writer, train_merged_sum, global_step=global_step)
                        epoch_step += 1
                        # run test if reach / 200 == 1
                        if (epoch_step * 200 % self.cfg['n_feature']) == 0 and epoch_step > 100:
                            # self.model.feed_data(sess, self.test[0], self.test[1])

                                # data, label = self.model.get_next(sess)
                                test_loss, test_pred = self.model.test(sess, batch_X, batch_y, current_lr)

                                print("Step:%d [Epoch:%d] [Learning rate: %.6f] train_loss:%.6f test_loss:%.6f" % (
                                    global_step, epoch_step, current_lr, train_loss, test_loss))

                                # Plot samples
                                # image_path = os.path.join(self.model_plots_dir,
                                #                           "SP500_epoch{:02d}_step{:04d}.png".format(epoch, epoch_step))
                                # sample_preds = test_pred[0]
                                # sample_truth = batch_y[0]
                                # self.plot_samples(sample_preds, sample_truth, image_path, stock_sym='SP500')

    def plot_samples(self, preds, targets, figname, stock_sym=None, multiplier=5):
        def _flatten(seq):
            return np.array([x for y in seq for x in y])

        print(targets)
        print(preds)
        truths = _flatten(targets)[-200:]
        preds = (_flatten(preds) * multiplier)[-200:]
        days = range(len(truths))[-200:]

        plt.figure(figsize=(12, 6))
        plt.plot(days, truths, label='truth')
        plt.plot(days, preds, label='pred')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("day")
        plt.ylabel("normalized price")
        plt.ylim((min(truths), max(truths)))
        plt.grid(ls='--')

        if stock_sym:
            plt.title(stock_sym + " | Last %d days in test" % len(truths))

        plt.savefig(figname, format='png', bbox_inches='tight', transparent=True)
        plt.close()

    @property
    def model_logs_dir(self):
        model_logs_dir = os.path.join(self.logs_dir, 'ma_modal')
        if not os.path.exists(model_logs_dir):
            os.makedirs(model_logs_dir)
        return model_logs_dir

    @property
    def model_plots_dir(self):
        model_plots_dir = os.path.join(self.plots_dir, 'ma_modal')
        if not os.path.exists(model_plots_dir):
            os.makedirs(model_plots_dir)
        return model_plots_dir

    @property
    def historical(self):
        return self._temperature

    @historical.setter
    def historical(self, value):
        self._temperature = value
