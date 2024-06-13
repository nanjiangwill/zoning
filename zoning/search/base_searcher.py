from abc import ABC, abstractmethod
from typing import List

from omegaconf import DictConfig

from zoning.class_types import EvaluationDatum, SearchResult


class Searcher(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def search(self, search_pattern: EvaluationDatum) -> List[SearchResult]:
        pass
