import flwr as fl

# Strategy: We'll use the default strategy for this example, which is FedAvg.
# Define strategy
# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=0.5,  # Fraction of clients to use for fit (default: 0.1)
#     min_fit_clients=3,  # Minimum number of clients to use for fit (default: 3)
#     min_available_clients=3,  # Minimum number of available clients (default: 3)
#     on_fit_config_fn=None,  # Function to configure on_fit (default: None)
#     on_evaluate_config_fn=None,  # Function to configure on_evaluate (default: None)
#     aggregator=None,  # Aggregator (default: None)
# )

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))