from class_types import IndexEntities
from omegaconf import DictConfig

from .base_indexer import Indexer


class EmbeddingIndexer(Indexer):
    def __init__(self, indexer_config: DictConfig):
        super().__init__(indexer_config)

    def index(self, index_entities: IndexEntities) -> None:
        pass
