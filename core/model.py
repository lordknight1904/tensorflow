import random

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# Third-party libraries
import numpy as np

import tensorflow as tf


x = tf.constant(-2.0, name="x", dtype=tf.float32)
a = tf.constant(5.0, name="a", dtype=tf.float32)
b = tf.constant(13.0, name="b", dtype=tf.float32)


y = tf.Variable(tf.add(tf.multiply(a, x), b))


init = tf.global_variables_initializer()


with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("../logs", session.graph)

    session.run(init)
    print(session.run(y))
