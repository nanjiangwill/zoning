from dataclasses import dataclass

from datasets import Dataset
from elasticsearch import Elasticsearch


@dataclass
class Indexer:
    def __init__(self, indexer_config):
        self.config = indexer_config

    def index(self, es: Elasticsearch, dataset: Dataset):
        raise NotImplementedError
