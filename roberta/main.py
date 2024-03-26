import flwr as fl
from typing import List
import torch
from collections import OrderedDict
from client import LoraClient
from peft import LoraConfig
# 创建客户端实例
def client_fn(cid: str) -> fl.client.Client:
    return LoraClient()

# 主程序
if __name__ == "__main__":
    # 定义 FedAvg 聚合策略
    model_cfg = {
    "model_name_or_path": "roberta-large",
    "peft_config": LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1),
}



    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.4,
        min_fit_clients=2,
        min_available_clients=5,
    )
    # server_address = "127.0.0.1:8080" 
    # 启动模拟，运行服务器和指定数量的客户端
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,  # 模拟的客户端数量
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        ray_init_args = {'num_cpus': 10, 'num_gpus': 1},
        client_resources={'num_cpus': 2, 'num_gpus': 0.5},
    )
    