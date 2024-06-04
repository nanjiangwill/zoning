from .base_indexer import Indexer
from omegaconf import DictConfig
from datasets import Dataset
from elasticsearch import Elasticsearch


class EmbeddingIndexer(Indexer):
    def __init__(self, indexer_config: DictConfig):
        super().__init__(indexer_config)

    def index(self, es: Elasticsearch, dataset: Dataset):
        pass
