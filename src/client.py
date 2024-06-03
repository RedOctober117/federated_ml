import tensorflow as tf
import flwr as fl
from flwr.server.strategy import FedAvg
import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import FeatureSpace
import sys

rng = np.random.default_rng()

def df_to_ds(df, target):
  dataframe = df.copy()
  labels = dataframe.pop(target)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  ds = ds.shuffle(buffer_size=len(dataframe))
  return ds

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

def slice_df(df, query):
  # Define Model
  indexes = df.query(query)
  # print(indexes)
  df = df.drop(indexes.index)
  return df.filter(['created', 'chargeTimeHrs', 'kwhTotal'])
  # print(data_df) 

# id = int(sys.argv[1])
# print(f'Establishing client using {id}')
path = 'normalized_ev_data_reduced.csv'

stations = []

with open('station_ids.txt', 'r') as file:
  for line in file.readlines():
    stations.append(line.strip('\n').strip())

rng.shuffle(stations)
i = 0
# with open('station_id_occurances.txt', 'r') as file:
#   for line in file.readlines():
#     clean = line.strip('\n').strip()
#     if clean not in stations:
#       stations[clean] = 1
#     else:
#       stations[clean] += 1

# for k, v in stations.items():
#   if v > 10:
#     print(k)

class ClientModel(fl.client.NumPyClient):
  def get_parameters(self, config):
    return self.model.get_weights()
  
  def fit(self, parameters, config):
    self.model.set_weights(parameters)
    self.model.fit(self.preprocessed_train_ds, epochs=10, validation_data=self.preprocessed_val_ds)
    return self.model.get_weights(), len(self.preprocessed_train_ds), {}
  
  def evaluate(self, parameters, config):
    self.model.set_weights(parameters)
    loss, accuracy = self.model.evaluate(self.preprocessed_train_ds)
    return loss, len(self.preprocessed_train_ds), {'accuracy': accuracy}
  
  def __init__(self, path, id):
    print(f'Creating model based on {id}')
    data_df = pd.read_csv(path)
    data_df = slice_df(data_df, str(f'stationId != {id}'), )

    val_df = data_df.sample(frac=0.1, random_state=1337)
    train_df = data_df.drop(val_df.index)

    print(f'Using {len(train_df)} samples for training, {len(val_df)} for validation.')

    train_ds = df_to_ds(train_df, 'kwhTotal')
    val_ds = df_to_ds(val_df, 'kwhTotal')

    train_ds = train_ds.batch((int(len(train_df) * 0.1)) + 1)
    val_ds = val_ds.batch((int(len(val_df) * 0.1)) + 1)


    feature_space = FeatureSpace(
      features={
        'chargeTimeHrs': 'float_normalized',
        'created': 'float_discretized',
      },

      output_mode='concat'
    )

    train_ds_no_labels = train_ds.map(lambda x, _: x)
    feature_space.adapt(train_ds_no_labels)

    print('Layers normalized...')

    preprocessed_train_ds = train_ds.map(
        lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    self.preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

    preprocessed_val_ds = val_ds.map(
        lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    self.preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

    print('Preprocessing layers prepped.')

    dict_inputs = feature_space.get_inputs()
    encoded_features = feature_space.get_encoded_features()

    x = keras.layers.Dense(100, activation='relu')(encoded_features)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(100, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(100, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    predictions = keras.layers.Dense(1, activation='sigmoid')(x)

    self.model = keras.Model(inputs=encoded_features, outputs=predictions)
    self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)



def client_fn(cid: str):
  global i
  return ClientModel(path, sys.argv[1:][0]).to_client()

# history = training_model.fit(
#     preprocessed_train_ds,
#     epochs=300,
#     validation_data=preprocessed_val_ds,
#     verbose=2,
# )

# epochs = history.epoch
# hist = pd.DataFrame(history.history)
# mse = hist['mean_squared_error']

# plot_the_loss_curve(epochs, mse, history.history["val_mean_squared_error"])

# Start
fl.client.start_client(
  server_address='127.0.0.1:8080',
  # Ignore "argument of type 'ClientModel'" error
  client=ClientModel(path, sys.argv[1:][0]).to_client(),
  max_retries=10000,
)

# NUM_CLIENTS=5

# strategy = FedAvg()

# client_resources = {"num_cpus": 4, "num_gpus": 0}

# fl.simulation.start_simulation(
#     client_fn=client_fn,
#     num_clients=NUM_CLIENTS,
#     config=fl.server.ServerConfig(num_rounds=5),
#     strategy=strategy,
#     client_resources=client_resources,
# )
