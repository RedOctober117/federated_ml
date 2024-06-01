import flwr as fl

strategy = fl.server.strategy.FedAvg()

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)



# hist = fl.simulation.start_simulation(
#   client_fn=client_fn,
#   num_clients=100,
#   config=fl.server.ServerConfig(num_rounds=10),
#   strategy=FedAvg(),
# )