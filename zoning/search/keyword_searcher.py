import json

from class_types import SearchPattern, SearchResult
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from omegaconf import DictConfig
from search.base_searcher import Searcher

# from ...utils import logger


class KeywordSearcher(Searcher):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.es_client = Elasticsearch(self.config.index.es_endpoint)
        self.num_results = self.config.search.num_results
        self.is_district_fuzzy = self.config.search.is_district_fuzzy
        self.is_eval_term_fuzzy = self.config.search.is_eval_term_fuzzy
        self.thesaurus_file = self.config.thesaurus_file

    def search(self, search_pattern: SearchPattern) -> list[SearchResult]:
        # Search in town
        s = Search(using=self.es_client, index=search_pattern.get_index_key())

        s.query = search_pattern.get_query()

        s = s.extra(size=self.num_results)

        s = s.highlight("Text")

        res = s.execute()
        if len(res) == 0:
            print(f"No results found for {search_pattern}")

        return [
            SearchResult(
                text=r.Text,
                page_number=r.Page,
                highlight=list(r.meta.highlight.Text),
                score=r.meta.score,
                query=json.dumps(s.query.to_dict()),
            )
            for r in res
        ]
