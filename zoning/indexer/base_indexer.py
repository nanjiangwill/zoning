from abc import ABC, abstractmethod

from datasets import Dataset
from elasticsearch import Elasticsearch


class Indexer(ABC):
    def __init__(self, indexer_config):
        self.config = indexer_config

    @abstractmethod
    def index(self, es: Elasticsearch, dataset: Dataset) -> None:
        pass
