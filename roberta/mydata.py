from flwr_datasets import FederatedDataset
from transformers import AutoTokenizer
import warnings


def get_fds(dataset, subset, client_num):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fds = FederatedDataset(dataset=dataset,subset=subset, partitioners={"train":client_num,"test":client_num})
    return fds

def get_tokenizer(model_name_or_path):
    padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def process_dataset(dataset,tokenizer):
    def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
            return outputs

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
        verbose=False,
    )

    dataset = dataset.rename_column("label", "labels")
    return dataset