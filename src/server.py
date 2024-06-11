from typing import Callable, Dict
import flwr as fl
import tensorflow as tf

# model = tf.keras.applications.EfficientNetB0(
#     input_shape=(32, 32, 3), weights=None, 
# )

fl.common.logger.configure(identifier='test_run', filename='log.txt')

def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(server_round: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "learning_rate": str(0.001),
            # "batch_size": str(32),
        }
        return config

    return fit_config

strategy = fl.server.strategy.FedAvg(
    # fraction_fit=0.1,
    # min_fit_clients=10,
    # min_available_clients=80,
    on_fit_config_fn=get_on_fit_config_fn(),
)

fed_trimmed_avg = fl.server.strategy.FedAvgM()

history = fl.server.start_server(
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
