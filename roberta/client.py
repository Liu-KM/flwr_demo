from collections import OrderedDict

import argparse
import os

import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from typing import Callable
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
import evaluate
from mydata import get_fds,get_tokenizer,process_dataset
from tqdm import tqdm
import flwr as fl
from model import get_model



model_cfg = {
    "model_name_or_path": "roberta-large",
    "peft_config": LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1),
}
batch_size = 32
model_name_or_path = "roberta-large"
task = "mrpc"
peft_type = PeftType.LORA
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_epochs =3

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
lr = 3e-4



    
def train(model,train_dataloader,epochs,device):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(epochs):
        model.to(device)
        model.train()
        total_loss = 0  # 初始化用于累加loss的变量
        num_batches = 0  # 记录处理的batch数量，用于计算平均loss
        for _, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            num_batches += 1
        # print(f"Epoch {epoch} - Average loss: {total_loss / num_batches}")


def test(model,testloader,metric,device):
    """Validate the network on the entire test set."""
    model.to(device)
    model.eval()
    
    for step, batch in enumerate(tqdm(testloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    eval_metric = metric.compute()
    # print(f"{eval_metric}")
    return eval_metric["f1"], eval_metric["accuracy"]



class LoraClient(fl.client.NumPyClient):
    def __init__(
    self,
    model_cfg,
    train_cfg,
    trainset,
    testset,
    model,
    tokenizer,
    ):  # pylint: disable=too-many-arguments
        self.model = model
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.metric = evaluate.load("glue", task)
        self.trainset = trainset
        self.testset = testset
        self.num_examples = {"trainset" : len(self.trainset), "testset" : len(self.testset)}
        self.tokenizer = tokenizer

    def get_parameters(self, config):
        """Return the parameters of the current net."""

        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def set_parameters(self, parameters) -> None:
        """Change the parameters of the model using the given ones."""
        peft_state_dict_keys = get_peft_model_state_dict(self.model).keys()
        params_dict = zip(peft_state_dict_keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(self.model, state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        def collate_fn(examples):
            return self.tokenizer.pad(examples, padding="longest", return_tensors="pt")
        train_dataloader = DataLoader(self.trainset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
        train(self.model,train_dataloader, epochs=1,device = self.device)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        def collate_fn(examples):
            return self.tokenizer.pad(examples, padding="longest", return_tensors="pt")
        eval_dataloader = DataLoader(
            self.testset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
        )
        loss, accuracy = test(self.model, eval_dataloader,self.metric,device = self.device)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def gen_client_fn(
    model_cfg,
    train_cfg,
    fds,
    model
) -> Callable[[str], LoraClient]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients."""

    def client_fn(cid: str) -> LoraClient:
        """Create a Flower client representing a single organization."""
        # Let's get the partition corresponding to the i-th client
        tokenizer = get_tokenizer(train_cfg.model_name_or_path)
        client_trainset = process_dataset(fds.load_partition(int(cid), "train"),tokenizer)
        client_testset = process_dataset(fds.load_partition(int(cid), "test"),tokenizer)
        
        return LoraClient(
            model_cfg,
            train_cfg,
            client_trainset,
            client_testset,
            model
            ).to_client()

    return client_fn
