from matplotlib.pylab import normal
import tensorflow as tf
import flwr as fl
from flwr.server.strategy import FedAvg
import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from flwr.common.logger import log
from logging import INFO, DEBUG, log
from matplotlib.backends.backend_pdf import PdfPages



# Source: https://stackoverflow.com/questions/49505872/read-json-to-pandas-dataframe-valueerror-mixing-dicts-with-non-series-may-lea
data = pd.read_csv('all_sample.csv')
retained_columns = ['datetime', 'I5-N VDS 759576']
data_df = data.loc[:, retained_columns]

normalized_df = pd.DataFrame()
scalar = MinMaxScaler()
normalized_df['datetime'] = pd.to_datetime(data_df['datetime'], format='%m/%d/%Y %H:%M')
normalized_df['I5-N VDS 759576'] = scalar.fit_transform(data_df['I5-N VDS 759576'].to_numpy().reshape(-1, 1))
print(normalized_df)
# normalized_df['I5-N VDS 759576'] = scalar.transform(normalized_df['I5-N VDS 759576'].to_numpy())

# normalized_df['kWhDelivered'] = (normalized_df['kWhDelivered'] - normalized_df['kWhDelivered'].mean()) / normalized_df['kWhDelivered'].std()
# normalized_df['kWhDelivered'] = 1 / (1 + math.e**(-1 * normalized_df['kWhDelivered']))
# normalized_df['kWhDelivered'] = (normalized_df['kWhDelivered'] - normalized_df['kWhDelivered'].min()) / (normalized_df['kWhDelivered'].max() - normalized_df['kWhDelivered'].min()) 

# normalized_df['connectionTime'] = pd.to_datetime(normalized_df['connectionTime'], format='%a, %d %b %Y %H:%M:%S %Z', utc=True)
# normalized_df['disconnectTime'] = pd.to_datetime(normalized_df['disconnectTime'], format='%a, %d %b %Y %H:%M:%S %Z', utc=True)

# normalized_df['chargeTime'] = (normalized_df['disconnectTime'] - normalized_df['connectionTime'])
# normalized_df['chargeTime'] = normalized_df['chargeTime'].seconds

# for line in range(len(normalized_df['chargeTime'])):
#   normalized_df.at[line, 'chargeTime'] = normalized_df.at[line, 'chargeTime'].seconds

# normalized_df['chargeTime'] = (normalized_df['chargeTime'] - normalized_df['chargeTime'].mean()) / normalized_df['chargeTime'].std()
# normalized_df['chargeTime'] = (normalized_df['chargeTime'] - normalized_df['chargeTime'].min()) / (normalized_df['chargeTime'].max() - normalized_df['chargeTime'].min()) 


# for line in range(len(normalized_df['chargeTime'])):
#   value = normalized_df.at[line, 'chargeTime']
#   normalized_df.at[line, 'chargeTime'] = (value * normalized_df['chargeTime'].min()) / (normalized_df['chargeTime'].max() - normalized_df['chargeTime'].min()) 
# normalized_df['chargeTime'] = (normalized_df['chargeTime'] - normalized_df['chargeTime'].mean()) / normalized_df['chargeTime'].std()
# normalized_df['chargeTime'] = 1 / (1 + math.e**(-1 * normalized_df['chargeTime']))

# for line in range(len(normalized_df['stationID'])):
  # normalized_df.at[line, 'connectionTime'] = datetime.datetime.strptime(normalized_df.at[line, 'connectionTime'], '%a, %d %b %Y %H:%M:%S %Z').date()
#   id = normalized_df.at[line, 'stationID'].split('-')
#   normalized_df.at[line, 'stationID'] = int(f'{id[2]}{id[3]}')

# normalized_df.sort_values(by=['stationID', 'connectionTime'], inplace=True)

# normalized_df.reset_index(drop=True, inplace=True)


print('\nDF mean:')
print(normalized_df.mean())
# print(normalized_df.hist())

plt.xlabel('observation')
plt.ylabel('traffic count')
plt.plot(normalized_df['I5-N VDS 759576'], label='traffic')
# plt.plot(normalized_df['connectionTime'][0:100], normalized_df['kWhDelivered'][0:100], label='kwh delivered')
plt.legend()
# plt.show()

# normalized_df = normalized_df.drop(normalized_df.query(str(f'stationID != 178817')).index)
# normalized_df_clean = normalized_df.drop(['connectionTime', 'disconnectTime', 'stationID'], axis=1)

# training_df = normalized_df.copy()
training_df = normalized_df[:int(len(normalized_df) * .7)]
test_df = normalized_df[int(len(normalized_df) * .7):]

# def df_to_ds(df, target):
#   dataframe = df.copy()
#   dataframe = dataframe.drop(['connectionTime', 'disconnectTime', 'stationID'], axis=1)
#   labels = dataframe.pop(target)
#   ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#   # ds = ds.shuffle(buffer_size=len(dataframe))
#   return ds

# print(training_df.columns)

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
# training_ds = df_to_ds(training_df, 'kWhDelivered')
# training_df_x = training_df.drop(index=0, axis=1)
training_df = training_df.pop('I5-N VDS 759576')
test_df = test_df.pop('I5-N VDS 759576')
training_x, training_y = split_sequence(training_df, steps)


# test_df_x = training_df.drop(index=0, axis=1)
# test_df_y = test_df.pop('kWhDelivered')
# val_ds = df_to_ds(val_df, 'kWhDelivered')
# test_ds = df_to_ds(test_df, 'kWhDelivered')

def train(x_train, y_train):
  model = keras.Sequential()
  model.add(keras.layers.LSTM(10, activation='relu', input_shape=(steps, 1)))
  # model.add(keras.layers.RepeatVector(1))
  # model.add(keras.layers.LSTM(100, activation='tanh', input_shape=(2, 1)))
  # model.add(keras.layers.RepeatVector(1))
  # model.add(keras.layers.LSTM(500, activation='tanh', input_shape=(3, 1)))
  # model.add(keras.layers.RepeatVector(1))
  # model.add(keras.layers.LSTM(100, activation='tanh', input_shape=(3, 1)))
  # model.add(keras.layers.RepeatVector(1))
  model.add(keras.layers.Dense(1, activation='linear'))
  model.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.MeanSquaredError()])
  history = model.fit(x_train, y_train, epochs=100, shuffle=False)

  return model, history

# model, history = train(training_df_x, training_df_y)

# def evaluate(model, data):
#   local_model, history = model
#   yhat = local_model.predict(data)
#   yhat = np.array(yhat).transpose(2, 0, 1).reshape(len(yhat), -1)

#   plt.xlabel('event index')
#   plt.ylabel('traffic flow')
#   plt.plot(normalized_df[:int(len(normalized_df) * .7)], label='true')
#   plt.plot(yhat, label='predicted')
#   # plt.xticks(np.arange(min(normalized_df_clean['kWhDelivered'][0]), max(normalized_df_clean['kWhDelivered'][0])+1, 2.0))
#   plt.legend()
#   plt.show()

# epochs = history.epoch
# hist = pd.DataFrame(history.history)
# mse_training = hist['mean_squared_error']
# mse_validation = history.history["val_mean_squared_error"]

#     epochs = self.history.epoch
#     hist = pd.DataFrame(self.history.history)
#     mse = hist['mean_squared_error']

#   """Plot a curve of loss vs. epoch."""

# plt.figure()
# plt.xlabel("Epoch")
# plt.ylabel("Mean Squared Error")
# plt.title(f'Epoch {time.time()} for station {id}')

# plt.plot(epochs, mse_training, label="Training Loss")
# plt.plot(epochs, mse_validation, label="Validation Loss")

# # mse_training is a pandas Series, so convert it to a list first.
# merged_mse_lists = mse_training.tolist() + mse_validation
# highest_loss = max(merged_mse_lists)
# lowest_loss = min(merged_mse_lists)
# top_of_y_axis = highest_loss * 1.03
# bottom_of_y_axis = lowest_loss * 0.97

# plt.ylim([bottom_of_y_axis, top_of_y_axis])
# plt.legend()
# plt.show()
# plt.clf()

#   global i
#   # pdf = PdfPages(f'figures\\{i}{id}_figures.pdf')
#   # pdf.savefig(f)
#   # pdf.close()
#   plt.savefig(f'figures\\{id}_{i}.png')
#   i += 1
#   plt.clf()


def round_based_learning(rounds=3):
  # local_models = [] 
  local_model_layers = []
  for round in range(rounds):
    model, history = train(training_x, training_y)
    local_model_layers.append(model)
    # local_models.append((model, history))

  # print(local_models)
  i = 1
  for model in local_model_layers:
    yhat = model.predict(test_df)
    # yhat = np.array(yhat).transpose(2, 0, 1).reshape(len(yhat), -1)

    plt.xlabel('events')
    plt.ylabel('traffic')
    plt.title(f'Model {i}')
    plt.plot(test_df, label='true')
    plt.plot(pd.DataFrame(yhat, index=test_df.index), label='predicted')
    plt.legend()
    plt.savefig(f'figures/model_{i}.png')
    plt.clf()
    i += 1



round_based_learning(2)


# IMPLEMENT THISSSSSSSSSS
# https://www.tensorflow.org/tutorials/structured_data/time_series#split_the_data

















# i = 1

# rng = np.random.default_rng()

# def df_to_ds(df, target):
#   dataframe = df.copy()
#   labels = dataframe.pop(target)
#   ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#   # ds = ds.shuffle(buffer_size=len(dataframe))
#   return ds

# def plot_the_loss_curve(id, epochs, mse_training, mse_validation):
#   """Plot a curve of loss vs. epoch."""

#   f = plt.figure()
#   plt.xlabel("Epoch")
#   plt.ylabel("Mean Squared Error")
#   plt.title(f'Epoch {time.time()} for station {id}')

#   plt.plot(epochs, mse_training, label="Training Loss")
#   plt.plot(epochs, mse_validation, label="Validation Loss")

#   # mse_training is a pandas Series, so convert it to a list first.
#   merged_mse_lists = mse_training.tolist() + mse_validation
#   highest_loss = max(merged_mse_lists)
#   lowest_loss = min(merged_mse_lists)
#   top_of_y_axis = highest_loss * 1.03
#   bottom_of_y_axis = lowest_loss * 0.97

#   plt.ylim([bottom_of_y_axis, top_of_y_axis])
#   plt.legend()
#   global i
#   # pdf = PdfPages(f'figures\\{i}{id}_figures.pdf')
#   # pdf.savefig(f)
#   # pdf.close()
#   plt.savefig(f'figures\\{id}_{i}.png')
#   i += 1
#   plt.clf()
  
#   # plt.show(block=True)

# def slice_df(df, query):
#   # Define Model
#   indexes = df.query(query)
#   # print(indexes)
#   df = df.drop(indexes.index)
#   return df.filter(['created', 'chargeTimeHrs', 'kwhTotal'])
#   # print(data_df) 

# # id = int(sys.argv[1])
# # print(f'Establishing client using {id}')
# path = 'normalized_ev_data_reduced.csv'

# stations = []

# with open('station_ids_greater_30.txt', 'r') as file:
#   for line in file.readlines():
#     stations.append(line.strip('\n').strip())

# class ClientModel(fl.client.NumPyClient):
#   def get_parameters(self, config):
#     return self.model.get_weights()
  
#   def fit(self, parameters, config):
#     self.model.set_weights(parameters)
#     self.history = self.model.fit(self.preprocessed_train_ds, epochs=600, validation_data=self.preprocessed_val_ds)

#     # self.plot_data.append((self.id, self.history.epoch, self.history.history['mean_squared_error'], self.history.history["val_mean_squared_error"]))
#     self.plot_loss()

#     return self.model.get_weights(), len(self.preprocessed_train_ds), {}
  
#   def evaluate(self, parameters, config):
#     self.model.set_weights(parameters)
#     self.loss, self.accuracy = self.model.evaluate(self.preprocessed_train_ds)

#     return self.loss, len(self.preprocessed_train_ds), {'accuracy': self.accuracy}
  
#   def __init__(self, path, id):
#     self.plot_data = []
#     self.id = id
#     log(INFO, f'Creating model based on {id}')
#     data_df = pd.read_csv(path)
#     data_df = slice_df(data_df, str(f'stationId != {id}'), )
#     data_df = data_df.drop('created', axis=1)
#     data_df['chargeTimeHrs'] = 1 / ( 1 + (math.e ** -data_df['chargeTimeHrs']) )

#     val_df = data_df.sample(frac=0.2, random_state=1337)
#     train_df = data_df.drop(val_df.index)

#     print(f'Using {len(train_df)} samples for training, {len(val_df)} for validation.')

#     train_ds = df_to_ds(train_df, 'kwhTotal')
#     val_ds = df_to_ds(val_df, 'kwhTotal')

#     train_ds = train_ds.batch((int(len(train_df) * 0.1)) + 1)
#     val_ds = val_ds.batch((int(len(val_df) * 0.1)) + 1)


    



#     train_ds_no_labels = train_ds.map(lambda x, _: x)
#     feature_space.adapt(train_ds_no_labels)

#     print('Layers normalized...')

#     preprocessed_train_ds = train_ds.map(
#         lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
#     )
#     self.preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

#     preprocessed_val_ds = val_ds.map(
#         lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
#     )
#     self.preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

#     print('Preprocessing layers prepped.')

#     dict_inputs = feature_space.get_inputs()
#     encoded_features = feature_space.get_encoded_features()

#     # https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
#     # https://towardsdatascience.com/a-practical-guide-to-rnn-and-lstm-in-keras-980f176271bc
#     # https://www.tensorflow.org/tutorials/structured_data/time_series

#     look_back = 1

#     model = keras.Sequential()
#     model.add(keras.LSTM(10, input_shape=(1, X_train_lmse.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
#     model.add(keras.Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
    
#     # x = keras.layers.Dense(10, activation='relu')(encoded_features)
#     # x = keras.layers.LSTM(1, activation='relu')(x)
#     # predictions = keras.layers.Dense(1, activation='linear')(x)

#     # model = keras.Sequential()
#     # model.add(keras.layers.LSTM(4))
    
#     # x = keras.layers.Dense(4, activation='relu')(encoded_features)
#     # # lstm = keras.layers.LSTM(4)
#     # # output = lstm(encoded_features)
#     # x = keras.layers.LSTM(4, return_sequences=True, return_state=True)
#     # predictions = keras.layers.Dense(1, activation='linear')(x)

#     # instead of SIGMOID

#     # self.model = keras.Model(inputs=encoded_features, outputs=predictions)
#     # self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.MeanSquaredError()])

#     # inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)
  
#   def plot_loss(self):
#     epochs = self.history.epoch
#     hist = pd.DataFrame(self.history.history)
#     mse = hist['mean_squared_error']

#     plot_the_loss_curve(self.id, epochs, mse, self.history.history["val_mean_squared_error"])





# def client_fn(cid: str):
#   return ClientModel(path, sys.argv[1:][0]).to_client()

# # model = ClientModel(path, sys.argv[1:][0])

# # history = training_model.fit(
# #     preprocessed_train_ds,
# #     epochs=300,
# #     validation_data=preprocessed_val_ds,
# #     verbose=2,
# # )



# # Start
# fl.client.start_client(
#   server_address='127.0.0.1:8080',
#   client=ClientModel(path, sys.argv[1:][0]).to_client(),
#   max_retries=10000,
# )



# # NUM_CLIENTS=5

# # strategy = FedAvg()

# # client_resources = {"num_cpus": 4, "num_gpus": 0}

# # fl.simulation.start_simulation(
# #     client_fn=client_fn,
# #     num_clients=NUM_CLIENTS,
# #     config=fl.server.ServerConfig(num_rounds=5),
# #     strategy=strategy,
# #     client_resources=client_resources,
# # )
