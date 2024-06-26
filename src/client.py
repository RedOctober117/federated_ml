from copy import deepcopy
import math
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
plt.xlabel('observation')
plt.ylabel('traffic count')
plt.plot(normalized_df['I5-N VDS 759576'], label='traffic')

plt.legend()
# plt.show()


training_df = normalized_df[:int(len(normalized_df) * .7)]
test_df = normalized_df[int(len(normalized_df) * .7):]

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
test_df = test_df.drop(columns='datetime')

clients = [
  (*split_sequence(training_df['I5-N VDS 759576'], steps), test_df.pop('I5-N VDS 759576')),
  (*split_sequence(training_df['I5-N VDS 763237'], steps), test_df.pop('I5-N VDS 763237')),
  (*split_sequence(training_df['I5-N VDS 759602'], steps), test_df.pop('I5-N VDS 759602')),
  (*split_sequence(training_df['I5-N VDS 716974'], steps), test_df.pop('I5-N VDS 716974')),
]

global_test = test_df.pop('I5-S VDS 71693')









def train(x_train, y_train, weights, epochs=100):
  model = keras.Sequential()
  if weights is not None:
    for layer_index in range(len(model.layers)):
        model.layers[layer_index].set_weights(weights[layer_index])

  model.add(keras.layers.LSTM(256, activation='relu', input_shape=(steps, 1), seed=1337, kernel_constraint=keras.constraints.NonNeg(), return_sequences=True))
  model.add(keras.layers.LSTM(64, activation='relu', seed=1337, kernel_constraint=keras.constraints.NonNeg()))
  model.add(keras.layers.Dense(1, activation='linear', kernel_constraint=keras.constraints.NonNeg()))
  model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanAbsoluteError()])
  history = model.fit(x_train, y_train, epochs=epochs, shuffle=False, verbose='1')

  return model, history







def federated_learning(clients, test_df, rounds=3, epochs=100) -> keras.models.Sequential:
  global_model = keras.Sequential()
  global_model.add(keras.layers.LSTM(256, activation='relu', input_shape=(steps, 1), seed=1337, kernel_constraint=keras.constraints.NonNeg(), return_sequences=True))
  global_model.add(keras.layers.LSTM(64, activation='relu', seed=1337, kernel_constraint=keras.constraints.NonNeg()))
  global_model.add(keras.layers.Dense(1, kernel_constraint=keras.constraints.NonNeg()))
  global_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanAbsoluteError()])
  
  client_model_history = []
  client_models = []
  prev_weights = None

  for round in range(rounds):
    print(f'\n\t### BEGINNING ROUND {round} ###\n')
    for client in clients:
      current_model, current_history = train(client[0], client[1], prev_weights, epochs=epochs)
      client_model_history.append(current_history)
      client_models.append(current_model)

    for layer_index in range(len(global_model.layers)):
      global_weights = global_model.layers[layer_index].get_weights()
      local_weights_list = [local_model.layers[layer_index].get_weights() for local_model in client_models]

      new_global_weights = []
      for weight_idx in range(len(global_weights)):
        local_weights_component = [local_weights[weight_idx] for local_weights in local_weights_list]

        averaged_weights_component = np.mean(local_weights_component, axis=0)
        new_global_weights.append(averaged_weights_component)

      global_model.layers[layer_index].set_weights(new_global_weights)
      prev_weights = new_global_weights.copy()

  i = 1
  for model, client, history in zip(client_models, clients, client_model_history):
    yhat = model.predict(client[2])
    plt.xlabel('events')
    plt.ylabel('traffic')
    plt.title(f'Model {i}')
    plt.plot(client[2], label='true')
    plt.plot(pd.DataFrame(yhat, index=client[2].index), label='predicted')
    plt.legend()
    plt.savefig(f'{path}/model_{i}')
    plt.clf()
    i += 1

  return global_model

round_count = 3
epoch_count = 200
model_layout = """
Local Models: LSTM 256 activation='relu', input_shape=(steps, 1), seed=1337, kernel_constraint=keras.constraints.NonNeg(), return_sequences=True; LSTM 64, activation='relu', seed=1337, kernel_constraint=keras.constraints.NonNeg(); Dense 1, activation='linear', kernel_constraint=keras.constraints.NonNeg()
Global Model: LSTM 256 activation='relu', input_shape=(steps, 1), seed=1337, kernel_constraint=keras.constraints.NonNeg(), return_sequences=True; LSTM 64, activation='relu', seed=1337, kernel_constraint=keras.constraints.NonNeg(); Dense 1, kernel_constraint=keras.constraints.NonNeg()
"""

model: keras.models.Sequential = federated_learning(clients, test_df, epochs=epoch_count, rounds=round_count)

yhat = model.predict(global_test)
# yhat = scalar.inverse_transform(yhat)
# global_test = scalar.inverse_transform(global_test.to_numpy().reshape(-1, 1))

plt.xlabel('events')
plt.ylabel('traffic')
plt.title(f'Global Model {time_}')
plt.plot(global_test, label='true')
plt.plot(pd.DataFrame(yhat, index=global_test.index), label='predicted')
plt.legend()
plt.savefig(f'{path.as_posix()}/global_model_{int(time_)}.png')
plt.clf()

logs = []

logs.append(f'Global Model: Rounds: {round_count} Epochs: {epoch_count}  Steps: {steps}\n Model description: {model_layout}\n')
logs.append(f'LSTM R2 score {sklearn.metrics.r2_score(global_test, yhat)}\n')
logs.append(f'LSTM MSE score {mean_squared_error(global_test, yhat)}\n')
logs.append(f'LSTM MAPE score {mean_absolute_percentage_error(global_test, yhat)}\n')
logs.append(f'LSTM MAE score {mean_absolute_error(global_test, yhat)}\n')
logs.append(f'LSTM MDAE score {median_absolute_error(global_test, yhat)}\n')
logs.append(f'LSTM RMSE score {math.sqrt(mean_squared_error(global_test, yhat))}\n')

with open(f'{path.as_posix()}/log.txt', 'w') as file:
  file.write(f'TIMESTAMP: {time_}\n')
  for log in logs:
    file.write(log)

