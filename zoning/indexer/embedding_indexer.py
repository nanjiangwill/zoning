from datasets import Dataset
from elasticsearch import Elasticsearch
from omegaconf import DictConfig

from .base_indexer import Indexer


class EmbeddingIndexer(Indexer):
    def __init__(self, indexer_config: DictConfig):
        super().__init__(indexer_config)

    def index(self, es: Elasticsearch, dataset: Dataset):
        pass
