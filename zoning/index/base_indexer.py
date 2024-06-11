from abc import ABC, abstractmethod

from class_types import IndexEntities
from omegaconf import DictConfig


class Indexer(ABC):
    def __init__(self, indexer_config: DictConfig):
        self.config = indexer_config

    @abstractmethod
    def index(self, index_entities: IndexEntities) -> None:
        pass
