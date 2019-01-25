import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as cPickle
import numpy as np
import gzip
import time

f = gzip.open('./data/mnist.pkl.gz', 'rb')
training_set, validation_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

training_label = np.zeros((training_set[1].size, training_set[1].max()+1))
training_label[np.arange(training_set[1].size), training_set[1]] = 1

validation_label = np.zeros((validation_set[1].size, validation_set[1].max()+1))
validation_label[np.arange(validation_set[1].size), validation_set[1]] = 1

# train = [[np.asarray(y) for y in x] for x in training_set[0]]
# print(training_set[0][0].shape)
# print(train[0].shape)

# model parameters
training_epochs = 25
learning_rate = 0.01
batch_size = 100
display_step = 1

datax = tf.placeholder(training_set[0].dtype, [None, 784])
labely = tf.placeholder(training_label.dtype, [None, 10])

training_data_set = tf.data.Dataset.from_tensor_slices((datax, labely)).batch(batch_size)

iterator = training_data_set.make_initializable_iterator()

# model
x = tf.placeholder("float", [None, 784], name='x')
y = tf.placeholder("float", [None, 10], name='y')

W = tf.Variable(tf.zeros([784, 10]), name='w')
b = tf.Variable(tf.zeros([10]), name='b')
activation = tf.nn.softmax(tf.matmul(x, W) + b)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(activation,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

cross_entropy = y * tf.log(activation)
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

avg_set = []
epoch_set = []
count = 0
next = iterator.get_next()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(training_epochs):
        # train the model
        count = 0
        sess.run(iterator.initializer, feed_dict={datax: training_set[0], labely: training_label})
        while True:
            try:
                data, label = sess.run(next)
                sess.run(optimizer, feed_dict={x: data, y: label})
                # print(sess.run(cost, feed_dict={x: data, y: label}))
                count += 1
                print("\rTraining epoch: {}({}/{})".format(epoch+1, count, int(len(training_set[0])/batch_size)), end="")
            except tf.errors.OutOfRangeError:
                break
        print("")

        # testing
        avg = []
        sess.run(iterator.initializer, feed_dict={datax: validation_set[0], labely: validation_label})
        while True:
            try:
                data, label = sess.run(next)
                avg.append(sess.run(accuracy, feed_dict={x: data, y: label}))
                print("\rAccuracy: {}".format(np.mean(avg)), end="")
            except tf.errors.OutOfRangeError:
                break
        print("")


