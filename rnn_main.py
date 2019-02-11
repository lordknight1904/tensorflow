import pickle as cPickle
import numpy as np
import gzip
import json
from core.RNN_model import RNN
from core.interface import Interface
import pandas as pd
import matplotlib.pyplot as plt
import json

# open config file
with open('./config.json') as json_data_file:
    cfg = json.load(json_data_file)

# getting data
raw_df = pd.read_csv('./data/SP500.csv')
size = cfg['n_step']

raw_seq = raw_df['Close'].tolist()
raw_seq = np.array(raw_seq)

seq = [np.array(raw_seq[i*cfg['n_feature']: (i+1)*cfg['n_feature']])
       for i in range(len(raw_seq) // cfg['n_feature'])]
seq = [seq[0] / seq[0][0] - 1.0] + [curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

X = np.array([seq[i: i + cfg['n_step']] for i in range(len(seq) - cfg['n_step'])])
y = np.array([seq[i + cfg['n_step']] for i in range(len(seq) - cfg['n_step'])])

train_size = int(len(X) * (1.0 - 0.05))
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

model = RNN(
    cfg
)
# model.dry_run(train_x, train_y)
#
interface = Interface(
    cfg=cfg,
    model=model,
    train=(train_X, train_y),
    test=(test_X, test_y),
)
# interface.historical = [y for x in validation_y for y in x]
interface.start()
