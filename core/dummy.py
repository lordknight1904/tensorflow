import pickle as cPickle
import gzip
import numpy as np
from tensorflow import layers
import tensorflow as tf


# if __name__ == "__main__":
#
#     inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor
#     x = layers.Dense(64, activation='relu')(inputs)
#     x = layers.Dense(64, activation='relu')(x)
#     predictions = layers.Dense(10, activation='softmax')(x)
#
#     model = tf.keras.Model(inputs=inputs, outputs=predictions)
#
#     model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=3.0),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     import numpy as np
#
#     data = np.random.random((1000, 32))
#     labels = np.random.random((1000, 10))
#
#     val_data = np.random.random((100, 32))
#     val_labels = np.random.random((100, 10))
#
#     model.fit(data, labels, epochs=10, batch_size=32,
#               validation_data=(val_data, val_labels))

def _parse_function(str):
    return str + "_"


if __name__ == "__main__":
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
    f.close()

    training_label = np.zeros((training_data[1].size, training_data[1].max()+1))
    training_label[np.arange(training_data[1].size), training_data[1]] = 1

    validation_label = np.zeros((validation_data[1].size, validation_data[1].max()+1))
    validation_label[np.arange(validation_data[1].size), validation_data[1]] = 1

    data_set = tf.data.Dataset.from_tensor_slices((training_data[0], training_label))
    data_set = data_set.batch(30).repeat()

    val_data_set = tf.data.Dataset.from_tensor_slices((validation_data[0], validation_label))
    val_data_set = val_data_set.batch(30).repeat()

    model = tf.keras.Sequential(
        [
            # layers.Dense(784, activation='sigmoid'),
            layers.Dense(30, activation='sigmoid', input_shape=(784,)),
            layers.Dense(10, activation='sigmoid')
        ]
    )

    # inputs = tf.keras.Input(shape=(784,))
    # x = layers.Dense(30, activation='sigmoid')(inputs)
    # predictions = layers.Dense(10, activation='sigmoid')(x)
    #
    # model = tf.keras.Model(inputs=inputs, outputs=predictions)

    model.summary()

    model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=3.0),
                  loss='mse',
                  metrics=['mae', 'accuracy'])

    model.fit(data_set, epochs=30, steps_per_epoch=50000, validation_data=val_data_set, validation_steps=3)
