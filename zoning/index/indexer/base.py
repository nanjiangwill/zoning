from datasets import Dataset


class Indexer:
    def __init__(self, indexer_config):
        self.config = indexer_config

    def index(self, data: Dataset):
        raise NotImplementedError
