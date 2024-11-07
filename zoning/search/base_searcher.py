from abc import ABC, abstractmethod

from zoning.class_types import SearchConfig, SearchQuery, SearchResult


class Searcher(ABC):
    def __init__(self, search_config: SearchConfig):
        self.search_config = search_config

    @abstractmethod
    def search(self, search_query: SearchQuery, target: str) -> SearchResult:
        pass
