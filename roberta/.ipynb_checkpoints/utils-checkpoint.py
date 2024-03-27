import flwr as fl
from flwr_datasets import FederatedDataset
from typing import Dict, Optional, Tuple
from mydata import get_tokenizer
from torch.utils.data import DataLoader
import torch
import datasets
from collections import OrderedDict
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
def get_evaluate_fn(model,logfile="log.csv"):
    """Return an evaluation function for server-side evaluation."""
    
    # Load data here to avoid the overhead of doing it in `evaluate` itself
    fds = FederatedDataset(dataset="glue",subset="mrpc", partitioners={"train": 10})
    test = fds.load_split("test")
    tokenizer = get_tokenizer("roberta-large")
    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs
    test = test.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
            # verbose=False,
        )
    test = test.rename_column("label", "labels")
    metric = datasets.load_metric("glue", "mrpc")
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        peft_state_dict_keys = get_peft_model_state_dict(model).keys()
        params_dict = zip(peft_state_dict_keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(model, state_dict)
        # model.set_weights(parameters)  # Update model with the latest parameters
        model.eval()
        def collate_fn(examples):
            return tokenizer.pad(examples, padding="longest", return_tensors="pt")
        testloader = DataLoader(test, shuffle=True, collate_fn=collate_fn, batch_size=16)
        total_loss = 0  # 初始化用于累加loss的变量
        num_batches = 0  # 记录处理的batch数量，用于计算平均loss
        for step, batch in enumerate(testloader):
            with torch.no_grad():
                outputs = model(**batch)
            
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
        loss = total_loss / num_batches
        eval_metric = metric.compute()
        f1, accuracy = eval_metric["f1"], eval_metric["accuracy"]
        print(f"Server evaluate result:\nRound: {server_round} \nAccuracy: {accuracy}, F1: {f1}")
        return loss, {"accuracy": accuracy, "f1": f1}

    return evaluate