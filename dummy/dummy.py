import pandas as pd
import matplotlib.pyplot as plt
import json

with open('./config.json') as json_data_file:
    cfg = json.load(json_data_file)

historical_data = pd.read_csv('./data/SP500.csv')
# historical_data = pd.read_csv('./data/SP500.csv', converters={"Adj Close": float})

print(historical_data['Adj Close'][:3])

plt.plot(historical_data['Adj Close'])
plt.show()

