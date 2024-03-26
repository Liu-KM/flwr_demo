import flwr as fl
from typing import List
import torch
from collections import OrderedDict
from client import gen_client_fn
from peft import LoraConfig,PeftType
from mydata import get_fds,get_tokenizer
from types import SimpleNamespace
import warnings
from utils import get_evaluate_fn
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # 加载模型的代码放在这里

# 主程序
if __name__ == "__main__":


    num_clients = 15
    # 定义 FedAvg 聚合策略
    model_cfg = {
    "model_name_or_path": "roberta-large",
    "peft_config": LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1),
    }
    model_cfg = SimpleNamespace(**model_cfg)
    train_cfg = {
    "max_length": 128,
    "batch_size": 32,
    "model_name_or_path": "roberta-large",
    "dataset": "glue",
    "task": "mrpc",
    "peft_type": PeftType.LORA,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "num_epochs": 3,
    "peft_config": LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1),
    "lr": 3e-4,
    }
    train_cfg = SimpleNamespace(**train_cfg)

    fds = get_fds(train_cfg.dataset,train_cfg.task,num_clients)
    
    tokenizer = get_tokenizer(train_cfg.model_name_or_path)

    client_fn = gen_client_fn(
        model_cfg,
        train_cfg,
        fds,
        tokenizer,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 加载模型的代码放在这里
        model = AutoModelForSequenceClassification.from_pretrained(model_cfg.model_name_or_path, return_dict=True)
        model = get_peft_model(model, model_cfg.peft_config)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.4,
        min_fit_clients=2,
        min_available_clients=num_clients,
        evaluate_fn=get_evaluate_fn(model)
    )

    # server_address = "127.0.0.1:8080" 
    # 启动模拟，运行服务器和指定数量的客户端
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,  # 模拟的客户端数量
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        ray_init_args = {'num_cpus': 80, 'num_gpus': 4},
        client_resources={'num_cpus': 2, 'num_gpus': 0.5},
    )
    