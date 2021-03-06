import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as cPickle
import numpy as np
import gzip

# getting data
f = gzip.open('../data/mnist.pkl.gz', 'rb')
training_set, validation_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

training_label = np.zeros((training_set[1].size, training_set[1].max()+1))
training_label[np.arange(training_set[1].size), training_set[1]] = 1

validation_label = np.zeros((validation_set[1].size, validation_set[1].max()+1))
validation_label[np.arange(validation_set[1].size), validation_set[1]] = 1

# model parameters
training_epochs = 25
learning_rate = 0.01
batch_size = 100
display_step = 1

datax = tf.placeholder(training_set[0].dtype, [None, 784])
labely = tf.placeholder(training_label.dtype, [None, 10])
print(training_label.dtype)
training_data_set = tf.data.Dataset.from_tensor_slices((datax, labely)).batch(batch_size)

iterator = training_data_set.make_initializable_iterator()

# model
_input = tf.placeholder("float", [None, 784], name='input')
prediction = tf.placeholder("float", [None, 10], name='prediction')

W = tf.Variable(tf.zeros([784, 10]), name='w')
b = tf.Variable(tf.zeros([10]), name='b')
activation = tf.nn.softmax(tf.matmul(_input, W) + b)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(activation, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

cross_entropy = prediction * tf.log(activation)
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

acc = []
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
                sess.run(optimizer, feed_dict={_input: data, prediction: label})
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
                a = sess.run(accuracy, feed_dict={_input: data, prediction: label})
                avg.append(a)
                print("\rAccuracy: {}".format(np.mean(avg)), end="")
            except tf.errors.OutOfRangeError:
                break
        print("")
        epoch_set.append(epoch + 1)
        acc.append(np.mean(avg))
    plt.plot(epoch_set, acc, label='Summary')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()


