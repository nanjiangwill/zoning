from omegaconf import DictConfig

from .base_indexer import Indexer
from .index_types import IndexEntities


class EmbeddingIndexer(Indexer):
    def __init__(self, indexer_config: DictConfig):
        super().__init__(indexer_config)

    def index(self, index_entities: IndexEntities) -> None:
        pass
