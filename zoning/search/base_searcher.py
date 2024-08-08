from abc import ABC, abstractmethod
from typing import Iterable

from zoning.class_types import SearchConfig, SearchQuery, SearchResult
from zoning.utils import get_thesaurus


class Searcher(ABC):
    def __init__(self, search_config: SearchConfig):
        self.search_config = search_config

    @abstractmethod
    def search(self, search_query: SearchQuery, target: str) -> SearchResult:
        pass
