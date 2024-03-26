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

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.4,  # 指定每轮用于训练的客户端的比例
    min_fit_clients=2,  # 指定每轮最少参与训练的客户端数量
    min_available_clients=5,  # 指定开始训练前需要的最少可用客户端数量
)

server_address = "127.0.0.1:8080" 
fl.server.start_server(
    server_address = server_address,
    config = fl.server.ServerConfig(num_rounds=10),
    strategy = strategy
)