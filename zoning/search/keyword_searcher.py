import json

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search
from omegaconf import DictConfig
from search.base_searcher import Searcher

from ..utils import District, PageSearchOutput, expand_term

# from ...utils import logger


class KeywordSearcher(Searcher):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.es_client = Elasticsearch(self.config.index.es_endpoint)
        self.num_results = self.config.search.num_results
        self.is_district_fuzzy = self.config.search.is_district_fuzzy
        self.is_term_fuzzy = self.config.search.is_term_fuzzy

    def search(self, town: str, district: District, term: str):
        print("Finally in")
        # Search in town
        s = Search(using=self.es_client, index=f"{self.config.target_state}-{town}")
        # Boost factor: Increasing the boost value will make documents matching this query to be ranked higher
        # Reference to Fuzzy: https://blog.mikemccandless.com/2011/03/lucenes-fuzzyquery-is-100-times-faster.html
        boost_value = 1.0

        exact_district_query = (
            Q("match_phrase", Text={"query": district.full_name, "boost": boost_value})
            | Q(
                "match_phrase",
                Text={"query": district.short_name, "boost": boost_value},
            )
            | Q(
                "match_phrase",
                Text={
                    "query": district.short_name.replace("-", ""),
                    "boost": boost_value,
                },
            )
            | Q(
                "match_phrase",
                Text={
                    "query": district.short_name.replace(".", ""),
                    "boost": boost_value,
                },
            )
        )

        fuzzy_district_query = Q(
            "match", Text={"query": district.short_name, "fuzziness": "AUTO"}
        ) | Q("match", Text={"query": district.full_name, "fuzziness": "AUTO"})

        if self.is_district_fuzzy:
            district_query = Q(
                "bool", should=[exact_district_query, fuzzy_district_query]
            )
        else:
            district_query = exact_district_query
        expanded_term = expand_term(self.config.thesaurus_file, term)
        exact_term_query = Q(
            "bool",
            should=list(Q("match_phrase", Text=t) for t in expanded_term),
            minimum_should_match=1,
        )

        if self.is_term_fuzzy:
            term_query = Q(
                "bool",
                should=[Q("match_phrase", Text=t) for t in expanded_term]
                + [
                    Q("match", Text={"query": t, "fuzziness": "AUTO"})
                    for t in expanded_term
                ],
                minimum_should_match=1,
            )
        else:
            term_query = exact_term_query
        units_expanded_term = expand_term(self.config.thesaurus_file, f"{term} units")
        units_query = Q(
            "bool",
            should=list(Q("match_phrase", Text=t) for t in units_expanded_term),
            minimum_should_match=1,
        )

        s.query = district_query & term_query & units_query

        # logger.info(f"Query: {s.query.to_dict()}")
        # ensure that we have a maximum of k results
        s = s.extra(size=self.num_results)

        s = s.highlight("Text")

        res = s.execute()
        if len(res) == 0:
            print(f"No results found for {term} in {town} {district.full_name}")

        yield from (
            PageSearchOutput(
                text=r.Text,
                page_number=r.Page,
                highlight=list(r.meta.highlight.Text),
                score=r.meta.score,
                query=json.dumps(s.query.to_dict()),
            )
            for r in res
        )
