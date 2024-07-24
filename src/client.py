from copy import deepcopy
import math
import random
import time
from matplotlib.pylab import normal
import sklearn
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, median_absolute_error
from sklearn.metrics import r2_score
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from flwr.common.logger import log
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf

time_ = int(time.time())

Path.mkdir(Path(Path.cwd(), 'figures', f'{time_}'))
path = Path(Path.cwd(), 'figures', f'{time_}')

data = pd.read_csv('ChargePoint Data CY20Q4.csv')
retained_columns = ['Station Name', 'Start Date', 'Energy (kWh)', 'Address 1']
data_df = data.loc[:, retained_columns]

data_df['Start Date'] = pd.to_datetime(data_df['Start Date'])
data_df['Start Date'] = data_df['Start Date'].dt.floor('D')
data_df.set_index('Start Date', inplace=True)

# tf_data = pd.read_csv('all_sample.csv')
# tf_retained_columns = ['datetime', 'I5-N VDS 759576', 'I5-N VDS 763237', 'I5-N VDS 759602', 'I5-N VDS 716974', 'I5-S VDS 71693']
# tf_data_df = tf_data.loc[:, tf_retained_columns]

# tf_data_df = yf.download("SPY", start="2017-01-01", end="2018-01-01")
# # print(tf_data_df.index)
# # plt.plot(tf_data_df['Close'])
# # plt.show(block=False)
# # plt.clf()
# tf_normalized_df = pd.DataFrame()
# tf_normalized_df.index = tf_data_df.index
# tf_scalar = MinMaxScaler(feature_range=(0,1))
# # tf_normalized_df['datetime'] = pd.to_datetime(tf_data_df['Date'], format='%m/%d/%Y %H:%M')
# tf_normalized_df['Close'] = tf_scalar.fit_transform(tf_data_df['Close'].to_numpy().reshape(-1, 1))

# tf_normalized_df = tf_normalized_df.pop('Close')
# tf_test_df_1 = tf_normalized_df[:int(len(tf_normalized_df) * .7)]
# tf_test_df_2 = tf_normalized_df[int(len(tf_normalized_df) * .7):]
# stations = {}
# for key in data_df['Station Name']:
#   if key in stations:
#     stations[key] += 1
#   else:
#     stations[key] = 1

# for key, value in stations.items():
#   print(key, value)

client_tables = {
  'MPL': [
    data_df[data_df['Station Name'] == 'PALO ALTO CA / MPL #4'].pop('Energy (kWh)'),
    data_df[data_df['Station Name'] == 'PALO ALTO CA / MPL #5'].pop('Energy (kWh)'),
    data_df[data_df['Station Name'] == 'PALO ALTO CA / MPL #6'].pop('Energy (kWh)'),
  ],
  'RINCONADA': [
    data_df[data_df['Station Name'] == 'PALO ALTO CA / RINCONADA LIB 1'].pop('Energy (kWh)'),
    data_df[data_df['Station Name'] == 'PALO ALTO CA / RINCONADA LIB 2'].pop('Energy (kWh)'),
    data_df[data_df['Station Name'] == 'PALO ALTO CA / RINCONADA LIB 3'].pop('Energy (kWh)'),
  ],
  'BRYANT': [
    data_df[data_df['Station Name'] == 'PALO ALTO CA / BRYANT #1'].pop('Energy (kWh)'),
    data_df[data_df['Station Name'] == 'PALO ALTO CA / BRYANT #2'].pop('Energy (kWh)'),
    data_df[data_df['Station Name'] == 'PALO ALTO CA / BRYANT #3'].pop('Energy (kWh)'),
    # data_df[data_df['Station Name'] == 'PALO ALTO CA / BRYANT #4'].pop('Energy (kWh)'),
    # data_df[data_df['Station Name'] == 'PALO ALTO CA / BRYANT #5'].pop('Energy (kWh)'),
    # data_df[data_df['Station Name'] == 'PALO ALTO CA / BRYANT #6'].pop('Energy (kWh)'),
  ],
  'HIGH': [
    data_df[data_df['Station Name'] == 'PALO ALTO CA / HIGH #1'].pop('Energy (kWh)'),
    data_df[data_df['Station Name'] == 'PALO ALTO CA / HIGH #2'].pop('Energy (kWh)'),
    data_df[data_df['Station Name'] == 'PALO ALTO CA / HIGH #3'].pop('Energy (kWh)'),
  ]
}


for key, clients in client_tables.items():
  temp_list = []
  for client in clients:
    fitted_client = client.resample('D', group_keys=True).sum()
    fitted_client = pd.DataFrame(ExponentialSmoothing(fitted_client, trend='add', seasonal='add', seasonal_periods=30).fit().fittedvalues)['2017-01-01':'2019-01-01']
    temp_list.append(fitted_client)
  client_tables[key] = temp_list

merged_clients = list()
for clients in client_tables.values():
  previous_merge = None
  i = 1
  for client in clients:
    if previous_merge is None:
      previous_merge = client
    else:
      previous_merge = pd.merge(previous_merge, client, how='outer', suffixes=(f'{i}', f'{i + 1}'), left_index=True, right_index=True)
      previous_merge.fillna(0, inplace=True)
      i += 1
  merged_clients.append(previous_merge)

def remove_outliers(items, quartiles, mean):
  new_array = list()
  for x in items:
    if x > quartiles[1] or x < quartiles[0]:
      print(f'REPLACING {x} WITH {mean}')
      new_array.append(mean)
    else:
      print(f'KEEPING x: {x}')
      new_array.append(x)

  return np.array(new_array)
    

scalar = MinMaxScaler(feature_range=(0,1))
normalized_clusters = list()
for cluster in merged_clients:
  normalized_clients = pd.DataFrame()
  normalized_clients.index = cluster.index 
  for client in cluster:
    # print(cluster[client].quantile(q=[0.25, 0.75]).to_numpy())
    # trimmed_client = pd.DataFrame(remove_outliers(cluster[client], cluster[client].quantile(q=[0.05, 0.95]).to_numpy(), cluster[client].mean()), index=cluster[client].index)
    normalized_clients[client] = scalar.fit_transform(cluster[client].to_numpy().reshape(-1, 1))
  normalized_clusters.append(normalized_clients)
# exit()
unobserved_test = data_df[data_df['Station Name'] == 'PALO ALTO CA / TED THOMPSON #1'].pop('Energy (kWh)')
fitted_unobserved_test = unobserved_test.resample('D', group_keys=True).sum()
fitted_unobserved_test = pd.DataFrame(ExponentialSmoothing(fitted_unobserved_test, trend='add', seasonal='add', seasonal_periods=30).fit().fittedvalues)['2017-01-01':'2019-01-01']
normalized_unobserved_test = pd.DataFrame()
normalized_unobserved_test.index = fitted_unobserved_test.index
normalized_unobserved_test[0] = scalar.fit_transform(fitted_unobserved_test.to_numpy().reshape(-1, 1))
normalized_unobserved_test_x = normalized_unobserved_test[:'2018-01-01']
normalized_unobserved_test_y = normalized_unobserved_test['2018-01-01':]


for cluster in normalized_clusters:
  print(cluster.info())
  print(cluster.describe(), end='\n\n\n')
# exit()

print(normalized_clusters[2].head())
# exit()
training_clusters = list()
testing_clusters = list()

for cluster in normalized_clusters:
  training_clusters.append(cluster[:'2018-01-01'])
  testing_clusters.append(cluster['2018-01-01':])

def sliding_window(df, features):
  x_out = list()
  y_out = list()

  for i in range(features + 1, len(df) + 1):
    x_out.append(df.iloc[i - (features + 1):i - 1])
    y_out.append([df.iloc[i - 1]])

  return np.array(x_out), np.array(y_out)

steps = 7

prepared_clusters = list()
for cluster_index in range(len(training_clusters)):
  prepared_cluster = list()
  for client in training_clusters[cluster_index]:
    prepared_cluster.append((*sliding_window(training_clusters[cluster_index][client], steps), *sliding_window(testing_clusters[cluster_index][client], steps)))
  prepared_clusters.append(prepared_cluster)

class Client():
  def __init__(self, x_train, y_train, x_test, y_test):
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test

    self.model = keras.Sequential()
    self.model.add(keras.layers.InputLayer((steps, 1)))
    self.model.add(keras.layers.LSTM(200, activation='relu', input_shape=(steps, 1)))
    self.model.add(keras.layers.RepeatVector(1))
    self.model.add(keras.layers.LSTM(200, activation='relu', input_shape=(steps, 1)))
    self.model.add(keras.layers.RepeatVector(1))
    self.model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
    self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
    self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
    self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
    self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear')))
    self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mean_absolute_error', metrics=[keras.metrics.MeanAbsoluteError()])

  def train(self, weights, epochs=100):
    print('Training. . .')
    if len(weights) > 0:
      for layer_index in range(len(self.model.layers)):
        self.model.layers[layer_index].set_weights(weights[layer_index])

    self.model.fit(self.x_train, self.y_train, epochs=epochs, shuffle=False)
  
  def evaluate(self, label, path):
    yhat = self.model.predict(self.x_test).reshape(-1, 1)

    plt.xlabel('events')
    plt.ylabel('traffic')
    plt.title(f'Model {label}')
    plt.plot(self.y_test, label='true')
    plt.plot(yhat, label='predicted')
    plt.legend()
    plt.savefig(f'{path}/model_{label}')
    plt.clf()
 
def plot_evaluation(yhat, actual, title, path, invert=True):
  if invert:
    yhat = scalar.inverse_transform(yhat.reshape(-1, 1))
    actual = pd.DataFrame(scalar.inverse_transform(actual.reshape(-1, 1)))

  plt.xlabel('events')
  plt.ylabel('Energy (kWh)')
  plt.title(f'{title}')
  plt.plot(actual, label='actual')
  plt.plot(yhat, label='predicted')
  plt.legend()
  plt.savefig(f'{path}')
  plt.clf()

  return yhat, actual

cluster_client_models = list()
for cluster in prepared_clusters:
  cluster_clients = list()
  for client in cluster:
    cluster_clients.append(Client(client[0], client[1], client[2], client[3]))
  cluster_client_models.append(cluster_clients)

def federated_learning(clients, test_df, rounds=3, epochs=100) -> keras.models.Sequential:
  global_model = keras.Sequential()
  global_model.add(keras.layers.InputLayer((steps, 1)))
  global_model.add(keras.layers.LSTM(200, activation='relu'))
  global_model.add(keras.layers.RepeatVector(1))
  global_model.add(keras.layers.LSTM(200, activation='relu'))
  global_model.add(keras.layers.RepeatVector(1))
  global_model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
  global_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
  global_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
  global_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
  global_model.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear')))
  global_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanAbsoluteError()])

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
    client.evaluate(i, f'{path.as_posix()}/cluster_{j}/')
    i += 1

  return global_model

round_count = 7
epoch_count = 200
model_layout = """
Client Model:
self.model = keras.Sequential()
self.model.add(keras.layers.InputLayer((steps, 1)))
self.model.add(keras.layers.LSTM(200, activation='relu', input_shape=(steps, 1)))
self.model.add(keras.layers.RepeatVector(1))
self.model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear')))
self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mean_absolute_error', metrics=[keras.metrics.MeanSquaredError()])
model.fit(self._x_train, self._y_train, epochs=epochs, shuffle=False)

Global Model:
global_model = keras.Sequential()
global_model.add(keras.layers.InputLayer((steps, 1)))
global_model.add(keras.layers.LSTM(200, activation='relu'))
global_model.add(keras.layers.RepeatVector(1))
global_model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
global_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
global_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
global_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
global_model.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear')))
global_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanSquaredError()])
"""
logs = []
logs.append(f'Model description: {model_layout}')

federated_models = list()
j = 1
for cluster in cluster_client_models:
  Path.mkdir(Path(path, f'cluster_{j}'))

  model = federated_learning(cluster, None, epochs=epoch_count, rounds=round_count)

  federated_models.append(model)

  i = 1
  logs.append(f'\n\t### CLUSTER {j} ###\n')

  for client in cluster:
    yhat, actual = plot_evaluation(model.predict(client.x_test), client.y_test, f'Global Model Observed Prediction {time_}', f'{path.as_posix()}/cluster_{j}/global_model_local_{i}_{int(time_)}.png')
    yhat = yhat.reshape(-1, 1)
    # global_test = client.x_test
    # yhat = model.predict(global_test)
    # yhat = scalar.inverse_transform(yhat)
    # global_test = scalar.inverse_transform(global_test)

    # plt.xlabel('events')
    # plt.ylabel('Energy (kWh)')
    # plt.title(f'Global Model Observed Prediction {time_}')
    # plt.plot(scalar.inverse_transform(client.y_test.reshape(-1, 1)), label='true')
    # plt.plot(yhat, label='predicted')
    # plt.legend()
    # plt.savefig(f'{path.as_posix()}/cluster_{j}/global_model_local_{i}_{int(time_)}.png')
    # plt.clf()

    logs.append(f'\t\n\t### GLOBAL ON CLIENT {i} ###\n')
    logs.append(f'\tGlobal Model: Rounds: {round_count} Epochs: {epoch_count}  Steps: {steps}\n')
    logs.append(f'\tLSTM R2 score {r2_score(actual, yhat)}\n')
    logs.append(f'\tLSTM MSE score {mean_squared_error(actual, yhat)}\n')
    logs.append(f'\tLSTM MAPE score {mean_absolute_percentage_error(actual, yhat)}\n')
    logs.append(f'\tLSTM MAE score {mean_absolute_error(actual, yhat)}\n')
    logs.append(f'\tLSTM MDAE score {median_absolute_error(actual, yhat)}\n')
    logs.append(f'\tLSTM RMSE score {math.sqrt(mean_squared_error(actual, yhat))}\n')
    i += 1

  with open(f'{path.as_posix()}/cluster_{j}/log.txt', 'w') as file:
    file.write(f'TIMESTAMP: {time_}\n')
    for log in logs:
      file.write(log)
  j += 1

# meta_model = keras.Sequential()
# meta_model.add(keras.layers.InputLayer((steps, 1)))
# meta_model.add(keras.layers.LSTM(200, activation='relu'))
# meta_model.add(keras.layers.RepeatVector(1))
# meta_model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
# meta_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
# meta_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
# meta_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
# meta_model.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear')))
# meta_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanAbsoluteError()])

# for layer_index in range(len(meta_model.layers)):
#   print('\n\t ### GATHERING WEIGHTS FOR META MODEL ###')

#   global_weights = meta_model.layers[layer_index].get_weights()
#   local_weights_list = [client.layers[layer_index].get_weights() for client in federated_models]

#   new_global_weights = []
#   for weight_idx in range(len(global_weights)):
#     local_weights_component = [local_weights[weight_idx] for local_weights in local_weights_list]
#     averaged_weights_component = np.mean(local_weights_component, axis=0)
#     new_global_weights.append(averaged_weights_component)

#   meta_model.layers[layer_index].set_weights(new_global_weights)

# for layer_index in range(len(meta_model.layers)):
#   print(meta_model.layers[layer_index].get_weights())

# loaded_meta_model = keras.saving.load_model(f'{path.as_posix()}/../../meta_model_{time_}.keras')

# yhat, actual = plot_evaluation(meta_model.predict(cluster_client_models[1][0].x_test).reshape(-1, 1), cluster_client_models[1][0].y_test, f'Meta Model Observed Prediction {time_}', f'{path.as_posix()}/meta_model_cluster_1_client_1_{int(time_)}.png', invert=False)
# print(yhat)
# yhat = yhat[:len(actual)]

# global_test_x = cluster_client_models[0][0].x_test
# global_test_y = cluster_client_models[0][0].y_test
# yhat = meta_model.predict(global_test_x)
# yhat = scalar.inverse_transform(yhat)
# global_test_y = pd.DataFrame(scalar.inverse_transform(global_test_y.reshape(-1, 1)))

# plt.xlabel('events')
# plt.ylabel('Energy (kWh)')
# plt.title(f'Meta Model Observed Prediction {time_}')
# plt.plot(global_test_y, label='true')
# plt.plot(yhat, label='predicted')
# plt.legend()
# plt.savefig(f'{path.as_posix()}/meta_model_cluster_1_client_1_{int(time_)}.png')
# plt.clf()


# logs = []
# model_layout = """
# Meta Model:
# meta_model = keras.Sequential()
# meta_model.add(keras.layers.InputLayer((steps, 1)))
# meta_model.add(keras.layers.LSTM(200, activation='relu'))
# meta_model.add(keras.layers.RepeatVector(1))
# meta_model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
# meta_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
# meta_model.add(keras.layers.TimeDistributed(keras.layers.Dense(16, activation='relu')))
# meta_model.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear')))
# meta_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanAbsoluteError()])
# """

# logs.append(f'Model description: {model_layout}')
# logs.append(f'\n\t### META MODEL OBSERVED ###\n')
# logs.append(f'LSTM R2 score {r2_score(actual, yhat)}\n')
# logs.append(f'LSTM MSE score {mean_squared_error(actual, yhat)}\n')
# logs.append(f'LSTM MAPE score {mean_absolute_percentage_error(actual, yhat)}\n')
# logs.append(f'LSTM MAE score {mean_absolute_error(actual, yhat)}\n')
# logs.append(f'LSTM MDAE score {median_absolute_error(actual, yhat)}\n')
# logs.append(f'LSTM RMSE score {math.sqrt(mean_squared_error(actual, yhat))}\n')

# yhat, actual = plot_evaluation(meta_model.predict(normalized_unobserved_test_x), normalized_unobserved_test_y[0], f'Meta Model Unobserved Prediction {time_}\n TED THOMPSON #1', f'{path.as_posix()}/meta_model_unobserved_client_{int(time_)}.png')

# global_test = normalized_unobserved_test_1
# yhat = meta_model.predict(global_test)
# yhat = scalar.inverse_transform(yhat)
# global_test = scalar.inverse_transform(global_test.to_numpy().reshape(-1, 1))

# plt.xlabel('events')
# plt.ylabel('Energy (kWh)')
# plt.title(f'Meta Model Unobserved Prediction {time_}\n TED THOMPSON #1')
# plt.plot(normalized_unobserved_test_2, label='true')
# plt.plot(yhat, label='predicted')
# plt.legend()
# plt.savefig(f'{path.as_posix()}/meta_model_unobserved_client_{int(time_)}.png')
# plt.clf()

# logs.append(f'\n\t### META MODEL UNOBSERVED ###\n')
# logs.append(f'LSTM R2 score {r2_score(actual, yhat)}\n')
# logs.append(f'LSTM MSE score {mean_squared_error(actual, yhat)}\n')
# logs.append(f'LSTM MAPE score {mean_absolute_percentage_error(actual, yhat)}\n')
# logs.append(f'LSTM MAE score {mean_absolute_error(actual, yhat)}\n')
# logs.append(f'LSTM MDAE score {median_absolute_error(actual, yhat)}\n')
# logs.append(f'LSTM RMSE score {math.sqrt(mean_squared_error(actual, yhat))}\n')

# global_test = tf_test_df_2
# yhat = meta_model.predict(tf_test_df_2)
# # yhat = tf_scalar.inverse_transform(yhat)
# # global_test = tf_scalar.inverse_transform(global_test.to_numpy().reshape(-1, 1))

# plt.xlabel('events')
# plt.ylabel('Energy (kWh)')
# plt.title(f'Meta Model Unobserved Prediction TF {time_}\n')
# plt.plot(global_test, label='true')
# plt.plot(pd.DataFrame(yhat, index=global_test.index), label='predicted')
# plt.legend()
# plt.savefig(f'{path.as_posix()}/meta_model_unobserved_tf_{int(time_)}.png')
# plt.clf()

# logs.append(f'\n\t### META MODEL UNOBSERVED ###\n')
# logs.append(f'LSTM R2 score {sklearn.metrics.r2_score(global_test, yhat)}\n')
# logs.append(f'LSTM MSE score {mean_squared_error(global_test, yhat)}\n')
# logs.append(f'LSTM MAPE score {mean_absolute_percentage_error(global_test, yhat)}\n')
# logs.append(f'LSTM MAE score {mean_absolute_error(global_test, yhat)}\n')
# logs.append(f'LSTM MDAE score {median_absolute_error(global_test, yhat)}\n')
# logs.append(f'LSTM RMSE score {math.sqrt(mean_squared_error(global_test, yhat))}\n')

# with open(f'{path.as_posix()}/meta_model_log.txt', 'w') as file:
#     file.write(f'TIMESTAMP: {time_}\n')
#     for log in logs:
#       file.write(log)

# meta_model.save(f'{path.as_posix()}/../../meta_model_{time_}.keras')