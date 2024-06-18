import json

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from tqdm.contrib.concurrent import thread_map

from zoning.class_types import (
    SearchConfig,
    SearchMatch,
    SearchQueries,
    SearchQuery,
    SearchResult,
    SearchResults,
)
from zoning.search.base_searcher import Searcher
from zoning.utils import get_district_query, get_eval_term_query, get_units_query


class KeywordSearcher(Searcher):
    def __init__(self, search_config: SearchConfig):
        super().__init__(search_config)
        self.es_client = Elasticsearch(self.search_config.es_endpoint)

    def _search(self, search_query: SearchQuery) -> SearchResult:
        try:
            s = Search(using=self.es_client, index=search_query.get_index_key())

            district_query = get_district_query(
                search_query.place.district_full_name,
                search_query.place.district_short_name,
                self.search_config.is_eval_term_fuzzy,
            )
            eval_term_query = get_eval_term_query(
                search_query.eval_term,
                self.search_config.is_eval_term_fuzzy,
                self.search_config.thesaurus_file,
            )
            units_query = get_units_query(
                search_query.eval_term, self.search_config.thesaurus_file
            )

            s.query = district_query & eval_term_query & units_query

            s = s.extra(size=self.search_config.num_results)

            s = s.highlight("Text")

            res = s.execute()
            if len(res) == 0:
                print(f"No results found for {search_query}")

            search_matches = [
                SearchMatch(
                    text=r.Text,
                    page_number=r.Page,
                    highlight=list(r.meta.highlight.Text),
                    score=r.meta.score,
                    query=json.dumps(s.query.to_dict()),
                )
                for r in res
            ]

            search_matches = sorted(search_matches, key=lambda x: x.score, reverse=True)

            return SearchResult(
                place=search_query.place,
                eval_term=search_query.eval_term,
                search_matches=search_matches,
            )

        except Exception as e:
            print(f"Error searching {search_query}")
            print(e)
            return None

    def search(self, search_queries: SearchQueries) -> SearchResults:
        search_results = thread_map(self._search, search_queries.search_queries)
        search_results = [sr for sr in search_results if sr]
        return SearchResults(search_results=search_results)
