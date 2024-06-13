import json
from typing import List

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from omegaconf import DictConfig
from search.base_searcher import Searcher

from zoning.class_types import EvaluationDatum, SearchResult
from zoning.utils import get_district_query, get_eval_term_query, get_units_query


class KeywordSearcher(Searcher):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.es_client = Elasticsearch(self.config.index.es_endpoint)
        self.num_results = self.config.search.num_results
        self.is_district_fuzzy = self.config.search.is_district_fuzzy
        self.is_eval_term_fuzzy = self.config.search.is_eval_term_fuzzy
        self.thesaurus_file = self.config.thesaurus_file

    def search(self, evaluation_datum: EvaluationDatum) -> List[SearchResult]:
        # Search in town
        s = Search(using=self.es_client, index=evaluation_datum.get_index_key())

        district_query = get_district_query(
            evaluation_datum.place.district_full_name,
            evaluation_datum.place.district_short_name,
            evaluation_datum.is_eval_term_fuzzy,
        )
        eval_term_query = get_eval_term_query(
            evaluation_datum.eval_term,
            evaluation_datum.is_eval_term_fuzzy,
            self.thesaurus_file,
        )
        units_query = get_units_query(evaluation_datum.eval_term, self.thesaurus_file)

        s.query = district_query & eval_term_query & units_query

        s = s.extra(size=self.num_results)

        s = s.highlight("Text")

        res = s.execute()
        if len(res) == 0:
            print(f"No results found for {evaluation_datum}")

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
