from .base import *
from omegaconf import DictConfig


class EmbeddingIndexer(Indexer):
    def __init__(self, indexer_config: DictConfig):
        super().__init__(indexer_config)

    def index(self, data: Dataset):
        pass
