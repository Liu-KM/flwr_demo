from collections import OrderedDict

import argparse
import os

import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
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
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
import flwr as fl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size = 32
model_name_or_path = "roberta-large"
task = "mrpc"
peft_type = PeftType.LORA
device = "cuda"
num_epochs = 20

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
lr = 3e-4
metric = evaluate.load("glue", task)


def load_data():
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset("glue", task)

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    num_examples = {"trainset" : len(tokenized_datasets["train"]), "testset" : len(tokenized_datasets["validation"])}
    return train_dataloader, eval_dataloader, num_examples
    
def train(model, train_dataloader,eval_dataloader, num_epochs):
    """Train the network on the training set."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
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
        print(f"epoch {epoch}:", eval_metric)

def test(net, testloader):
    """Validate the network on the entire test set."""
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
    print(f"{eval_metric}")
    return eval_metric["f1"], eval_metric["accuracy"]

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

trainloader, testloader, num_examples = load_data()

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return {
            k: v.cpu()
            for k, v in get_peft_model_state_dict(model).items()
        }

    def set_parameters(self, weights):
        return set_peft_model_state_dict(model, weights)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model, trainloader,testloader, epochs=1)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(model, testloader)
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}
    
fl.client.start_client(server_address="[::]:8080", client=CifarClient().to_client())