from abc import ABC, abstractmethod

from zoning.class_types import SearchConfig, SearchQueries, SearchResults


class Searcher(ABC):
    def __init__(self, search_config: SearchConfig):
        self.search_config = search_config

    @abstractmethod
    def search(self, search_queries: SearchQueries) -> SearchResults:
        pass
