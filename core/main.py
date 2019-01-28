import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as cPickle
import numpy as np
import gzip
import json
from core.model import Model
from core.interface import Interface

# open config file
with open('./config.json') as json_data_file:
    cfg = json.load(json_data_file)

# getting data
f = gzip.open('../data/mnist.pkl.gz', 'rb')
training_set, validation_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

training_label = np.zeros((training_set[1].size, training_set[1].max()+1))
training_label[np.arange(training_set[1].size), training_set[1]] = 1

validation_label = np.zeros((validation_set[1].size, validation_set[1].max()+1))
validation_label[np.arange(validation_set[1].size), validation_set[1]] = 1

model = Model(cfg, {"dtype": training_set[0].dtype, "size": 784}, {"dtype": training_label.dtype, "size": 10})
interface = Interface(
    model=model,
    train=(training_set[0], training_label),
    validation=(validation_set[0], validation_label),
)
interface.start()
