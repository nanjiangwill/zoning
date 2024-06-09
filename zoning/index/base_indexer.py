from abc import ABC, abstractmethod

from omegaconf import DictConfig

from .index_types import IndexEntities


class Indexer(ABC):
    def __init__(self, indexer_config: DictConfig):
        self.config = indexer_config

    @abstractmethod
    def index(self, index_entities: IndexEntities) -> None:
        pass
