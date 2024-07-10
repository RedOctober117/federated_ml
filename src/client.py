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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

time_ = int(time.time())

Path.mkdir(Path(Path.cwd(), 'figures', f'{time_}'))
path = Path(Path.cwd(), 'figures', f'{time_}')



data = pd.read_csv('ChargePoint Data CY20Q4.csv')
retained_columns = ['Station Name', 'Start Date', 'Energy (kWh)', 'Address 1']
data_df = data.loc[:, retained_columns]


data_df['Start Date'] = pd.to_datetime(data_df['Start Date'])
data_df['Start Date'] = data_df['Start Date'].dt.floor('D')
data_df.set_index('Start Date', inplace=True)
# data_df.resample('D').sum().sort_index(inplace=True)
# data_df.sort_values(by=['Station Name', ], inplace=True)
# print(data_df)

plt.ylabel('kWh')
plt.plot(data_df['Energy (kWh)'], label='observed')
plt.legend()
plt.clf()
# plt.show()
# print(data_df)

stations = {}
for key in data_df['Station Name']:
  if key in stations:
    stations[key] += 1
  else:
    stations[key] = 1

for key, value in stations.items():
  print(key, value)

# client_1 = data_df[data_df['Station Name'] == 'PALO ALTO CA / MPL #3'].pop('Energy (kWh)')
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
    data_df[data_df['Station Name'] == 'PALO ALTO CA / BRYANT #4'].pop('Energy (kWh)'),
    data_df[data_df['Station Name'] == 'PALO ALTO CA / BRYANT #5'].pop('Energy (kWh)'),
    data_df[data_df['Station Name'] == 'PALO ALTO CA / BRYANT #6'].pop('Energy (kWh)'),
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
    fitted_client = pd.DataFrame(ExponentialSmoothing(fitted_client, trend='add', seasonal='add', seasonal_periods=30).fit().fittedvalues)['2018-01-01':'2020-01-01']
    temp_list.append(fitted_client)
  client_tables[key] = temp_list

print(client_tables)
# exit()
# client_2 = client_2.resample('D', group_keys=True).sum()
# client_3 = client_3.resample('D', group_keys=True).sum()
# client_4 = client_4.resample('D', group_keys=True).sum()

# client_2 = pd.DataFrame(ExponentialSmoothing(client_2, trend='add', seasonal='add', seasonal_periods=30).fit().fittedvalues)['2014-09-24':'2016-09-24']
# client_3 = pd.DataFrame(ExponentialSmoothing(client_3, trend='add', seasonal='add', seasonal_periods=30).fit().fittedvalues)['2014-09-24':'2016-09-24']
# client_4 = pd.DataFrame(ExponentialSmoothing(client_4, trend='add', seasonal='add', seasonal_periods=30).fit().fittedvalues)['2014-09-24':'2016-09-24']

# plt.plot(client_2)
# plt.legend()
# plt.show()
# plt.clf()
# decomposed_client_2 = seasonal_decompose(client_4, period=30)
# decomposed_client_2.plot().show()
# print(client_1)
# print(client_2)
# print(client_3)
# print(client_4)

# merged_data_df_1 = pd.merge(client_1, client_2, how='outer', suffixes=('_1', '_2'), left_index=True, right_index=True)
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
# merged_data_df_2 = pd.merge(client_3, client_4, how='outer', suffixes=('_1', '_2'), left_index=True, right_index=True)
# merged_data_df = pd.merge(client_2, merged_data_df_2,  how='outer', suffixes=('_1', '_2'), left_index=True, right_index=True)
# merged_data_df.fillna(0, inplace=True)
print(merged_clients)

scalar = MinMaxScaler(feature_range=(0,1))
normalized_clusters = list()
for cluster in merged_clients:
  normalized_clients = pd.DataFrame()
  normalized_clients.index = cluster.index 
  for client in cluster:
    normalized_clients[client] = scalar.fit_transform(cluster[client].to_numpy().reshape(-1, 1))
  normalized_clusters.append(normalized_clients)

print(normalized_clusters)

# normalized_df = pd.DataFrame()
# normalized_df.index = merged_data_df.index
# normalized_df['Energy (kWh)'] = scalar.fit_transform(merged_data_df[0].to_numpy().reshape(-1, 1))
# normalized_df['Energy (kWh)_1'] = scalar.fit_transform(merged_data_df['0_1'].to_numpy().reshape(-1, 1))
# normalized_df['Energy (kWh)_2'] = scalar.fit_transform(merged_data_df['0_2'].to_numpy().reshape(-1, 1))
# normalized_df['Energy (kWh)_2_2'] = scalar.fit_transform(merged_data_df['Energy (kWh)_2_2'].to_numpy().reshape(-1, 1))

# print(normalized_df['Energy (kWh)'])
# plt.xlabel('hour')
# plt.ylabel('Energy')
# plt.title('Client 1')
# plt.plot(normalized_df['Energy (kWh)'], label='client_1')

# plt.legend()
# # plt.show()

# plt.clf()


# plt.xlabel('hour')
# plt.ylabel('Energy')
# plt.title('Client 2')


# plt.legend()
# plt.show()
# plt.clf()

for cluster in normalized_clusters:
  print(cluster.info(), end='\n\n\n')

training_clusters = list()
testing_clusters = list()

for cluster in normalized_clusters:
  training_clusters.append(cluster[:'2019-01-01'])
  testing_clusters.append(cluster['2019-01-01':])

# training_df = normalized_df[:'2016-01-01']
# test_df = normalized_df['2016-01-01':]

def split_sequence(sequence, n_steps):
  X, y = list(), list()
  for i in range(len(sequence)):
    # find the end of this pattern
    end_ix = i + n_steps
    # check if we are beyond the sequence
    if end_ix > len(sequence)-1:
      break
    # gather input and output parts of the pattern
    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
    X.append(seq_x)
    y.append(seq_y)
  return np.array(X), np.array(y)

steps = 1

prepared_clusters = list()

print(training_clusters)

for cluster_index in range(len(normalized_clusters)):
  prepared_cluster = list()
  for client in normalized_clusters[cluster_index]:
    prepared_cluster.append((*split_sequence(normalized_clusters[cluster_index][client], steps), testing_clusters[cluster_index][client]))
  prepared_clusters.append(prepared_cluster)

print(prepared_clusters)
# clients = [
  # (*split_sequence(training_df['Energy (kWh)'], steps), test_df.pop('Energy (kWh)')),
  # (*split_sequence(training_df['Energy (kWh)_1'], steps), test_df.pop('Energy (kWh)_1')),
  # (*split_sequence(training_df['Energy (kWh)_2'], steps), test_df.pop('Energy (kWh)_2')),
  # (*split_sequence(training_df['Energy (kWh)_2_2'], steps), test_df.pop('Energy (kWh)_2_2')),
  # (*split_sequence(training_df['I5-N VDS 716974'], steps), test_df.pop('I5-N VDS 716974')),
# ]

# print('CLIENT DATA: ', clients[0][2], end='\n\n')
# print(clients[0][1], end='\n\n')
# print(clients[0][2], end='\n\n')
# exit()
# print(clients[0][1])
# print(clients[0][2])
# exit()
# global_test = test_df.pop('')

# IDEA: Gather a single sensor and split data into 4 equal sets for 4 clients. 
# Test global model on both the original total data set and then on each 
# individual client set.
class Client():
  def __init__(self, x_train, y_train, test_df):
    self._x_train = x_train
    self._y_train = y_train
    self.test_df = test_df

    self.model = keras.Sequential()
    self.model.add(keras.layers.InputLayer((steps, 1)))
    self.model.add(keras.layers.LSTM(units=64, kernel_constraint=keras.constraints.NonNeg()))
    self.model.add(keras.layers.Dense(units=8, activation='relu', kernel_constraint=keras.constraints.NonNeg()))
    self.model.add(keras.layers.Dense(units=1, activation='linear', kernel_constraint=keras.constraints.NonNeg()))
    self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mean_absolute_error', metrics=[keras.metrics.MeanSquaredError()])

  def train(self, weights=None, epochs=100):
    print('Training. . .')
    if weights is not None:
      for layer_index in range(len(self.model.layers)):
        self.model.layers[layer_index].set_weights(weights[layer_index])

    self.model.fit(self._x_train, self._y_train, epochs=epochs, shuffle=False)
  
  def evaluate(self, label, path):
    yhat = self.model.predict(self.test_df)
    plt.xlabel('events')
    plt.ylabel('traffic')
    plt.title(f'Model {label}')
    plt.plot(self.test_df, label='true')
    plt.plot(pd.DataFrame(yhat, index=self.test_df.index), label='predicted')
    plt.legend()
    plt.savefig(f'{path}/model_{label}')
    # plt.show()
    plt.clf()



# client_models = [ Client(client[0], client[1], client[2]) for client in clients ]
cluster_client_models = list()
for cluster in prepared_clusters:
  cluster_clients = list()
  for client in cluster:
    cluster_clients.append(Client(client[0], client[1], client[2]))
  cluster_client_models.append(cluster_clients)

print(cluster_client_models)
print(cluster_client_models[0])
print(cluster_client_models[0][0])
print(cluster_client_models[0][0].test_df)

def federated_learning(clients, test_df, rounds=3, epochs=100) -> keras.models.Sequential:
  global_model = keras.Sequential()
  global_model.add(keras.layers.InputLayer((steps, 1)))
  global_model.add(keras.layers.LSTM(units=64, kernel_constraint=keras.constraints.NonNeg()))
  global_model.add(keras.layers.Dense(units=8, activation='relu', kernel_constraint=keras.constraints.NonNeg()))
  global_model.add(keras.layers.Dense(units=1, activation='linear', kernel_constraint=keras.constraints.NonNeg()))
  global_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanSquaredError()])

  weights = None

  for round in range(rounds):
    for client in clients:
      print(f'\n\t### ROUND {round}: Training client. . . ###\n')
      client.train(weights, epochs)

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
  weights = new_global_weights

  i = 1
  for client in clients:
    client.evaluate(i, f'{path.as_posix()}/cluster_{j}/')
    i += 1

  return global_model

round_count = 5
epoch_count = 20
model_layout = """
Client Model:
model = keras.Sequential()
self.model.add(keras.layers.InputLayer((steps, 1)))
self.model.add(keras.layers.LSTM(units=200,  return_sequences=True))
self.model.add(keras.layers.LSTM(units=64,  ))
self.model.add(keras.layers.Dense(units=8, activation='relu',  ))
self.model.add(keras.layers.Dense(units=1, activation='linear',  ))
self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mean_absolute_error', metrics=[keras.metrics.MeanSquaredError()])
model.fit(self._x_train, self._y_train, epochs=epochs, shuffle=False)

Global Model:
global_model.add(keras.layers.InputLayer((steps, 1)))
global_model.add(keras.layers.LSTM(units=200,  return_sequences=True))
global_model.add(keras.layers.LSTM(units=64,  ))
global_model.add(keras.layers.Dense(units=8, activation='relu',  ))
global_model.add(keras.layers.Dense(units=1, activation='linear',  ))
global_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanSquaredError()])
"""
logs = []
logs.append(f'Model description: {model_layout}')

federated_models = list()
j = 1
for cluster in cluster_client_models:
  Path.mkdir(Path(path, f'cluster_{j}'))
  # path = Path(Path.cwd(), 'figures', f'{time_}', f'cluster_{j}')
  model = federated_learning(cluster, None, epochs=epoch_count, rounds=round_count)
  federated_models.append(model)
  i = 1
  logs.append(f'\n\t### CLUSTER {j} ###\n')
  for client in cluster:
    global_test = client.test_df
    yhat = model.predict(global_test)
    yhat = scalar.inverse_transform(yhat)
    global_test = scalar.inverse_transform(global_test.to_numpy().reshape(-1, 1))

    plt.xlabel('events')
    plt.ylabel('Energy (kWh)')
    plt.title(f'Global Model Observed Prediction {time_}')
    plt.plot(global_test, label='true')
    plt.plot(yhat, label='predicted')
    plt.legend()
    plt.savefig(f'{path.as_posix()}/cluster_{j}/global_model_local_{i}_{int(time_)}.png')
    plt.clf()

    logs.append(f'\t\n\t### CLIENT {i} ###\n')
    logs.append(f'\tGlobal Model: Rounds: {round_count} Epochs: {epoch_count}  Steps: {steps}\n')
    logs.append(f'\tLSTM R2 score {sklearn.metrics.r2_score(global_test, yhat)}\n')
    logs.append(f'\tLSTM MSE score {mean_squared_error(global_test, yhat)}\n')
    logs.append(f'\tLSTM MAPE score {mean_absolute_percentage_error(global_test, yhat)}\n')
    logs.append(f'\tLSTM MAE score {mean_absolute_error(global_test, yhat)}\n')
    logs.append(f'\tLSTM MDAE score {median_absolute_error(global_test, yhat)}\n')
    logs.append(f'\tLSTM RMSE score {math.sqrt(mean_squared_error(global_test, yhat))}\n')
    i += 1

  with open(f'{path.as_posix()}/cluster_{j}/log.txt', 'w') as file:
    file.write(f'TIMESTAMP: {time_}\n')
    for log in logs:
      file.write(log)
  j += 1
# model: keras.models.Sequential = federated_learning(client_models, None, epochs=epoch_count, rounds=round_count)


# global_test = client[2]
# yhat = model.predict(global_test)
# yhat = scalar.inverse_transform(yhat)
# global_test = scalar.inverse_transform(global_test.to_numpy().reshape(-1, 1))

# plt.xlabel('events')
# plt.ylabel('traffic')
# plt.title(f'Global Model Unobserved Prediction {time_}')
# plt.plot(global_test, label='true')
# plt.plot(yhat, label='predicted')
# plt.legend()
# plt.savefig(f'{path.as_posix()}/global_model_global_{int(time_)}.png')
# plt.clf()

# logs.append(f'\n\t### GLOBAL TEST ###\n')
# logs.append(f'Global Model: Rounds: {round_count} Epochs: {epoch_count}  Steps: {steps}\n')
# logs.append(f'LSTM R2 score {sklearn.metrics.r2_score(global_test, yhat)}\n')
# logs.append(f'LSTM MSE score {mean_squared_error(global_test, yhat)}\n')
# logs.append(f'LSTM MAPE score {mean_absolute_percentage_error(global_test, yhat)}\n')
# logs.append(f'LSTM MAE score {mean_absolute_error(global_test, yhat)}\n')
# logs.append(f'LSTM MDAE score {median_absolute_error(global_test, yhat)}\n')
# logs.append(f'LSTM RMSE score {math.sqrt(mean_squared_error(global_test, yhat))}\n')



meta_model = keras.Sequential()
meta_model.add(keras.layers.InputLayer((steps, 1)))
meta_model.add(keras.layers.LSTM(units=64, kernel_constraint=keras.constraints.NonNeg()))
meta_model.add(keras.layers.Dense(units=8, activation='relu', kernel_constraint=keras.constraints.NonNeg()))
meta_model.add(keras.layers.Dense(units=1, activation='linear', kernel_constraint=keras.constraints.NonNeg()))
meta_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanSquaredError()])

# weights = None

for layer_index in range(len(meta_model.layers)):
  print('\n\t ### GATHERING WEIGHTS FOR META MODEL ###')
  # print(f'\n\t### ROUND {round}: Gathering weights for layer {layer_index}. . . ###\n')
  global_weights = meta_model.layers[layer_index].get_weights()
  local_weights_list = [client.layers[layer_index].get_weights() for client in federated_models]

  new_global_weights = []
  for weight_idx in range(len(global_weights)):
    # print(f'\n\t### ROUND {round}: Averaging weights for layer {weight_idx}. . . ###\n')
    local_weights_component = [local_weights[weight_idx] for local_weights in local_weights_list]

    averaged_weights_component = np.mean(local_weights_component, axis=0)
    new_global_weights.append(averaged_weights_component)

  # print(f'\n\t### ROUND {round}: Updating weights for global model layer {layer_index}. . . ###\n')
  meta_model.layers[layer_index].set_weights(new_global_weights)


global_test = cluster_client_models[0][0].test_df
yhat = meta_model.predict(global_test)
yhat = scalar.inverse_transform(yhat)
global_test = scalar.inverse_transform(global_test.to_numpy().reshape(-1, 1))

plt.xlabel('events')
plt.ylabel('Energy (kWh)')
plt.title(f'Meta Model Observed Prediction {time_}')
plt.plot(global_test, label='true')
plt.plot(yhat, label='predicted')
plt.legend()
plt.savefig(f'{path.as_posix()}/meta_model_cluster_1_client_1_{int(time_)}.png')
plt.clf()
