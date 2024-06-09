from abc import ABC, abstractmethod

from class_types import SearchPattern, SearchResult
from omegaconf import DictConfig


class Searcher(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def search(self, search_pattern: SearchPattern) -> list[SearchResult]:
        pass
