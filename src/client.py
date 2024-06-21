from matplotlib.pylab import normal
import tensorflow as tf
from flwr.server.strategy import FedAvg
import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from flwr.common.logger import log
from logging import INFO, DEBUG, log
from matplotlib.backends.backend_pdf import PdfPages

def plot_the_loss_curve(epochs, mse_training, mse_validation):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")

  plt.plot(epochs, mse_training, label="Training Loss")
  plt.plot(epochs, mse_validation, label="Validation Loss")

  # mse_training is a pandas Series, so convert it to a list first.
  merged_mse_lists = mse_training.tolist() + mse_validation
  highest_loss = max(merged_mse_lists)
  lowest_loss = min(merged_mse_lists)
  top_of_y_axis = highest_loss * 1.03
  bottom_of_y_axis = lowest_loss * 0.97

  plt.ylim([bottom_of_y_axis, top_of_y_axis])
  plt.legend()
  plt.show()

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
# print(normalized_df.hist())

plt.xlabel('observation')
plt.ylabel('traffic count')
plt.plot(normalized_df['I5-N VDS 759576'], label='traffic')
# plt.plot(normalized_df['I5-N VDS 763237'], label='traffic')
# plt.plot(normalized_df['I5-N VDS 759602'], label='traffic')
# plt.plot(normalized_df['I5-N VDS 716974'], label='traffic')
plt.legend()
plt.show()


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

def train(x_train, y_train, epochs=100):
  model = keras.Sequential()
  model.add(keras.layers.LSTM(10, activation='relu', input_shape=(steps, 1)))
  model.add(keras.layers.Dense(1, activation='linear'))
  model.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.MeanSquaredError()])
  history = model.fit(x_train, y_train, epochs=epochs, shuffle=False)

  return model, history


def round_based_learning(training_x, training_y, rounds=3):
  local_model_layers = []
  for round in range(rounds):
    model, history = train(training_x, training_y)
    local_model_layers.append(model)

  i = 1
  for model in local_model_layers:
    yhat = model.predict(test_df)

    plt.xlabel('events')
    plt.ylabel('traffic')
    plt.title(f'Model {i}')
    plt.plot(test_df, label='true')
    plt.plot(pd.DataFrame(yhat, index=test_df.index), label='predicted')
    plt.legend()
    plt.savefig(f'figures/model_{i}.png')
    plt.clf()
    i += 1

def federated_learning(clients, test_df, rounds=3, epochs=100) -> keras.models.Sequential:
  global_model: keras.models.Sequential = keras.Sequential()
  global_model.add(keras.layers.Dense(1, activation='linear'))
  client_model_history = []
  client_models = []
  print(len(clients[0]))
  for round in range(rounds):
    print(f'\n\t### BEGINNING ROUND {round} ###\n')
    for client in clients:
      current_model, current_history = train(client[0], client[1], epochs)
      client_model_history.append(current_history)
      client_models.append(current_model)

    if global_model is None:
      global_model = client_models[0]
    else:
        for layer_index in range(len(global_model.layers)):
            global_weights = global_model.layers[layer_index].get_weights()
            local_weights_list = [local_model.layers[layer_index].get_weights() for local_model in client_models]

            new_global_weights = []
            for weight_idx in range(len(global_weights)):
                local_weights_component = [local_weights[weight_idx] for local_weights in local_weights_list]

                averaged_weights_component = np.mean(local_weights_component, axis=0)
                new_global_weights.append(averaged_weights_component)

            global_model.layers[layer_index].set_weights(new_global_weights)

  i = 1
  for model, client, history in zip(client_models, clients, client_model_history):
    yhat = model.predict(client[2])
    plt.xlabel('events')
    plt.ylabel('traffic')
    plt.title(f'Model {i}, Round {round}')
    plt.plot(client[2], label='true')
    plt.plot(pd.DataFrame(yhat, index=test_df.index), label='predicted')
    plt.legend()
    plt.savefig(f'figures/model_{i}_{round}.png')
    plt.clf()

    # epochs = history.epoch
    # hist = pd.DataFrame(history.history)
    # mse = hist['mean_squared_error']

    # plot_the_loss_curve(epochs, mse, history.history['mean_squared_error'])

    # plt.xlabel('epochs')
    # plt.ylabel('MSE')
    # plt.title(f'Model {i}, Round {round} MSE')
    # plt.plot(epochs, mse, label='Training Loss')
    # # plt.plot(epochs, history.history["val_mean_squared_error"], label='Validation Loss')
    # plt.legend()
    # plt.savefig(f'figures/model_loss_{i}_{round}.png')
    # plt.clf()
    i += 1

  return global_model

model: keras.models.Sequential = federated_learning(clients, test_df, epochs=200)

# model.add(keras.layers.Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.MeanSquaredError()])
yhat = model.predict(global_test)
# yhat = model.predict(clients[0][2])
plt.xlabel('events')
plt.ylabel('traffic')
plt.title(f'Global Model')
plt.plot(global_test, label='true')
plt.plot(pd.DataFrame(yhat, index=test_df.index), label='predicted')
plt.legend()
plt.savefig(f'figures/global_model.png')
plt.clf()