import flwr as fl
import tensorflow as tf

# model = tf.keras.applications.EfficientNetB0(
#     input_shape=(32, 32, 3), weights=None, 
# )

fl.common.logger.configure(identifier='test_run', filename='log.txt')

strategy = fl.server.strategy.FedAvg(
    # fraction_fit=0.1,
    # min_fit_clients=5,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
)



# hist = fl.simulation.start_simulation(
#   client_fn=client_fn,
#   num_clients=100,
#   config=fl.server.ServerConfig(num_rounds=10),
#   strategy=FedAvg(),
# )