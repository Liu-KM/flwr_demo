from flwr_datasets import FederatedDataset
from transformers import AutoTokenizer


def get_fds(dataset, subset, client_num):
    fds = FederatedDataset(dataset=dataset,subset=subset, partitioners={"train":client_num,"test":client_num})
    return fds

def get_tokenizer(model_name_or_path):
    padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer