from copy import deepcopy
import math
import random
import time
from matplotlib.pylab import normal
import sklearn
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, median_absolute_error
import sklearn.metrics
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from flwr.common.logger import log
from pathlib import Path

time_ = int(time.time())

Path.mkdir(Path(Path.cwd(), 'figures', f'{time_}'))
path = Path(Path.cwd(), 'figures', f'{time_}')



data = pd.read_csv('all_sample.csv')
retained_columns = ['datetime', 'I5-N VDS 759576', 'I5-N VDS 763237', 'I5-N VDS 759602', 'I5-N VDS 716974', 'I5-S VDS 71693']
data_df = data.loc[:, retained_columns]

normalized_df = pd.DataFrame()
scalar = MinMaxScaler(feature_range=(0,1))
normalized_df['datetime'] = pd.to_datetime(data_df['datetime'], format='%m/%d/%Y %H:%M')
normalized_df['I5-N VDS 759576'] = scalar.fit_transform(data_df['I5-N VDS 759576'].to_numpy().reshape(-1, 1))
normalized_df['I5-N VDS 763237'] = scalar.fit_transform(data_df['I5-N VDS 763237'].to_numpy().reshape(-1, 1))
normalized_df['I5-N VDS 759602'] = scalar.fit_transform(data_df['I5-N VDS 759602'].to_numpy().reshape(-1, 1))
normalized_df['I5-N VDS 716974'] = scalar.fit_transform(data_df['I5-N VDS 716974'].to_numpy().reshape(-1, 1))
normalized_df['I5-S VDS 71693'] = scalar.fit_transform(data_df['I5-S VDS 71693'].to_numpy().reshape(-1, 1))
print(normalized_df)

print('\nDF mean:')
print(normalized_df.mean())
plt.xlabel('hour')
plt.ylabel('traffic count')
plt.plot(normalized_df['I5-N VDS 759576'], label='traffic')

plt.legend()
# plt.show()
plt.clf()


training_df = normalized_df[:int(len(normalized_df) * .7)]
test_df = normalized_df[int(len(normalized_df) * .7):]

def sliding_window(df, features):
  x_out = list()
  y_out = list()

  for i in range(features + 1, len(df) + 1):
    x_out.append(df.iloc[i - (features + 1):i - 1])
    y_out.append([df.iloc[i - 1]])

  return np.array(x_out), np.array(y_out)

steps = 24
test_df = test_df.drop(columns='datetime')

clients = [
  (*sliding_window(training_df['I5-N VDS 759576'], steps), *sliding_window(test_df.pop('I5-N VDS 759576'), steps)),
  (*sliding_window(training_df['I5-N VDS 763237'], steps), *sliding_window(test_df.pop('I5-N VDS 763237'), steps)),
  (*sliding_window(training_df['I5-N VDS 759602'], steps), *sliding_window(test_df.pop('I5-N VDS 759602'), steps)),
  (*sliding_window(training_df['I5-N VDS 716974'], steps), *sliding_window(test_df.pop('I5-N VDS 716974'), steps)),
]

global_test_x, global_test_y = sliding_window(test_df.pop('I5-S VDS 71693'), steps)

# IDEA: Gather a single sensor and split data into 4 equal sets for 4 clients. 
# Test global model on both the original total data set and then on each 
# individual client set.

class Client():
  def __init__(self, x_train, y_train, x_test, y_test):
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test

    self.model = keras.Sequential()
    self.model.add(keras.layers.LSTM(200, activation='relu', input_shape=(steps, 1)))
    self.model.add(keras.layers.RepeatVector(1))
    self.model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
    self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
    self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
    self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear')))
    self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mean_absolute_error', metrics=[keras.metrics.MeanSquaredError()])

  def train(self, weights, epochs=100):
    print('Training. . .')
    if len(weights) is not 0:
      for layer_index in range(len(self.model.layers)):
        self.model.layers[layer_index].set_weights(weights[layer_index])

    self.model.fit(self.x_train, self.y_train, epochs=epochs, shuffle=False, verbose='3', )
  
  def evaluate(self, label):
    yhat = self.model.predict(self.x_test).reshape(-1, 1)
    plt.xlabel('events')
    plt.ylabel('traffic')
    plt.title(f'Model {label}')
    plt.plot(self.y_test, label='true')
    plt.plot(yhat, label='predicted')
    plt.legend()
    plt.savefig(f'{path}/model_{label}')
    plt.clf()



client_models = [ Client(client[0], client[1], client[2], client[3]) for client in clients ]



def federated_learning(clients, test_df, rounds=3, epochs=100) -> keras.models.Sequential:
  global_model = keras.Sequential()
  global_model.add(keras.layers.LSTM(200, activation='relu', input_shape=(steps, 1)))
  global_model.add(keras.layers.RepeatVector(1))
  global_model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
  global_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
  global_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
  global_model.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear')))
  global_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanSquaredError()])

  weights = list()

  for round in range(rounds):
    for client in clients:
      print(f'\n\t### ROUND {round}: Training client. . . ###\n')
      client.train(weights, epochs)

    weights.clear()

    for layer_index in range(len(global_model.layers)):
      print(f'\n\t### ROUND {round}: Gathering weights for layer {layer_index}. . . ###\n')
      global_weights = global_model.layers[layer_index].get_weights()
      local_weights_list = [client.model.layers[layer_index].get_weights() for client in clients]

      new_global_weights = []
      for weight_idx in range(len(global_weights)):
        print(f'\n\t### ROUND {round}: Averaging weights for layer {weight_idx}. . . ###\n')
        local_weights_component = [local_weights[weight_idx] for local_weights in local_weights_list]

        averaged_weights_component = np.mean(local_weights_component, axis=0)
        new_global_weights.append(averaged_weights_component)

      print(f'\n\t### ROUND {round}: Updating weights for global model layer {layer_index}. . . ###\n')
      global_model.layers[layer_index].set_weights(new_global_weights)
      weights.append(new_global_weights)

  i = 1
  for client in clients:
    client.evaluate(i)
    i += 1

  return global_model

round_count = 20
epoch_count = 50
model_layout = """
Client Model:
model = keras.Sequential()
model.add(keras.layers.InputLayer((steps, 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(units=8, activation='relu'))
model.add(keras.layers.Dense(units=1, activation='linear'))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mean_absolute_error', metrics=[keras.metrics.MeanSquaredError()])
model.fit(self._x_train, self._y_train, epochs=epochs, shuffle=False, verbose='3', )

Global Model:
global_model.add(keras.layers.InputLayer((steps, 1)))
global_model.add(keras.layers.LSTM(units=64))  # global_model.add(keras.layers.LSTM(1, seed=1337))
global_model.add(keras.layers.Dense(units=8, activation='relu'))
global_model.add(keras.layers.Dense(units=1, activation='linear'))
global_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanSquaredError()])
"""

model: keras.models.Sequential = federated_learning(client_models, None, epochs=epoch_count, rounds=round_count)

logs = []
logs.append(f'Model description: {model_layout}')

# global_test = client[2]
global_test = global_test_y
yhat = model.predict(global_test_x).reshape(-1, 1)
# yhat = scalar.inverse_transform(yhat)
# global_test = scalar.inverse_transform(global_test_y)

plt.xlabel('events')
plt.ylabel('traffic')
plt.title(f'Global Model Unobserved Prediction {time_}')
plt.plot(global_test, label='true')
plt.plot(yhat, label='predicted')
plt.legend()
plt.savefig(f'{path.as_posix()}/global_model_global_{int(time_)}.png')
plt.clf()

logs.append(f'\n\t### GLOBAL TEST ###\n')
logs.append(f'Global Model: Rounds: {round_count} Epochs: {epoch_count}  Steps: {steps}\n')
logs.append(f'LSTM R2 score {sklearn.metrics.r2_score(global_test, yhat)}\n')
logs.append(f'LSTM MSE score {mean_squared_error(global_test, yhat)}\n')
logs.append(f'LSTM MAPE score {mean_absolute_percentage_error(global_test, yhat)}\n')
logs.append(f'LSTM MAE score {mean_absolute_error(global_test, yhat)}\n')
logs.append(f'LSTM MDAE score {median_absolute_error(global_test, yhat)}\n')
logs.append(f'LSTM RMSE score {math.sqrt(mean_squared_error(global_test, yhat))}\n')

i = 1
for client in client_models:
  actual = client.y_test
  yhat = model.predict(client.x_test).reshape(-1, 1)
  # yhat = scalar.inverse_transform(yhat)
  # actual = scalar.inverse_transform(actual)

  plt.xlabel('events')
  plt.ylabel('traffic')
  plt.title(f'Global Model Observed Prediction {time_}')
  plt.plot(actual, label='true')
  plt.plot(yhat, label='predicted')
  plt.legend()
  plt.savefig(f'{path.as_posix()}/global_model_local_{i}_{int(time_)}.png')
  plt.clf()

  logs.append(f'\n\t### CLIENT {i} ###\n')
  logs.append(f'Global Model: Rounds: {round_count} Epochs: {epoch_count}  Steps: {steps}\n')
  logs.append(f'LSTM R2 score {sklearn.metrics.r2_score(actual, yhat)}\n')
  logs.append(f'LSTM MSE score {mean_squared_error(actual, yhat)}\n')
  logs.append(f'LSTM MAPE score {mean_absolute_percentage_error(actual, yhat)}\n')
  logs.append(f'LSTM MAE score {mean_absolute_error(actual, yhat)}\n')
  logs.append(f'LSTM MDAE score {median_absolute_error(actual, yhat)}\n')
  logs.append(f'LSTM RMSE score {math.sqrt(mean_squared_error(actual, yhat))}\n')
  i += 1

with open(f'{path.as_posix()}/log.txt', 'w') as file:
  file.write(f'TIMESTAMP: {time_}\n')
  for log in logs:
    file.write(log)

