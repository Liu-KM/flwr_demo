from flwr_datasets import FederatedDataset
from transformers import AutoTokenizer


def get_fds(dataset, subset, client_num):
    fds = FederatedDataset(dataset=dataset,subset=subset, partitioners={"train":client_num})
    return fds