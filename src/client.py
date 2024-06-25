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
scalar = MinMaxScaler()
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
  # for layer_index in range(len(model.layers)):
  #     model.layers[layer_index].set_weights(weights[layer_index])
  # if weights is not None:
    # model.set_weights(weights)
  model.add(keras.layers.LSTM(100, activation='relu', input_shape=(steps, 1), seed=1337))
  model.add(keras.layers.Dense(1, activation='linear'))
  model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanAbsoluteError()])
  history = model.fit(x_train, y_train, epochs=epochs, shuffle=False)

  return model, history


# def round_based_learning(training_x, training_y, rounds=3):
#   local_model_layers = []
#   for round in range(rounds):
#     model, history = train(training_x, training_y)
#     local_model_layers.append(model)

#   i = 1
#   for model in local_model_layers:
#     yhat = model.predict(test_df)

#     plt.xlabel('events')
#     plt.ylabel('traffic')
#     plt.title(f'Model {i}')
#     plt.plot(test_df, label='true')
#     plt.plot(pd.DataFrame(yhat, index=test_df.index), label='predicted')
#     plt.legend()
#     plt.savefig(f'figures/model_{i}.png')
#     plt.clf()
#     i += 1

def federated_learning(clients, test_df, rounds=3, epochs=100) -> keras.models.Sequential:
  global_model: keras.models.Sequential = keras.Sequential()
  client_model_history = []
  client_models = []
  prev_weights = None
  # PSEUCODE and visualize
  for round in range(rounds):
    print(f'\n\t### BEGINNING ROUND {round} ###\n')
    for client in clients:
      current_model, current_history = train(client[0], client[1], prev_weights, epochs=epochs)
      client_model_history.append(current_history)
      client_models.append(current_model)

    if global_model is None:
      global_model = client_models[0]
    else:
      local_weights = []
      for layer in range(len(global_model.layers)):
        local_weights = [ model.get_weights()[layer] for model in client_models ]

        avg_weights = np.mean(local_weights, axis=0)

        # for global_layer_index in range(len(global_model.layers)):
        global_model.layers[layer].set_weights(avg_weights)
      
      # prev_weights = [ np.array(weight) for weight in avg_weights ]

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
    print(model.layers)
    print(len(model.layers))
    i += 1

  return global_model

round_count = 3
epoch_count = 100
model_layout = 'LSTM: 100 relu mean_absolute_error, \n Dense: 1 linear mean_absolute_error'

model: keras.models.Sequential = federated_learning(clients, test_df, epochs=epoch_count, rounds=round_count)
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.MeanAbsoluteError()])
yhat = model.predict(global_test)

plt.xlabel('events')
plt.ylabel('traffic')
plt.title(f'Global Model: R: {round_count} E: {epoch_count} M: {model_layout}')
plt.plot(global_test, label='true')
plt.plot(pd.DataFrame(yhat, index=global_test.index), label='predicted')
plt.legend()
plt.savefig(f'{path.as_posix()}/global_model_{int(time_)}.png')
plt.clf()

logs = []

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

print(model.layers)
print(len(model.layers))

print(model.get_weights())
