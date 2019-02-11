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
# with open('./config.json') as json_data_file:
#     cfg = json.load(json_data_file)
#
# # getting data
# historical_data = pd.read_csv('./data/SP500.csv')
# size = cfg['n_step']
#
# historical_data = historical_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']].values
# last = 0
# # normalize
# for i in range(len(historical_data)-1, -1, -1):
#     if i-size > -1:
#         historical_data[i][3] = historical_data[i][3] / historical_data[i-size][3] - 1.
#     else:
#         last = i
#         break
# historical_data[:last, 3] = 0.
#
# checkpoint = round(len(historical_data)*0.9)
# train = historical_data[:round(len(historical_data)*0.9)]
# validation = historical_data[round(len(historical_data)*0.9)-len(historical_data):]
#
# train = [train[i*size: i*size+size] for i in range(int(len(train)/size))]
# # train_x = [np.delete(a, 3, 1) for a in train]
# train_x = [np.delete(a, 3, 1) for a in train]
# train_y = [a[:, 3] for a in train]
#
# validation = [validation[i*size: i*size+size] for i in range(int(len(validation)/size))]
# validation_x = [np.delete(a, 3, 1) for a in validation]
# validation_y = [a[:, 3] for a in validation]

# model = RNN(
#     cfg
# )
# model.dry_run(train_x, train_y)
#
# interface = Interface(
#     cfg=cfg,
#     model=model,
#     train=(train_x, train_y),
#     validation=(train_x, train_y),
# )
# interface.historical = [y for x in validation_y for y in x]
# interface.start()

list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
assert all(x > 1 for x in list), 'Error'
