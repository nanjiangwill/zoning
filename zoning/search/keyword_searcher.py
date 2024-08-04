import json

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

from zoning.class_types import SearchConfig, SearchMatch, SearchQuery, SearchResult
from zoning.search.base_searcher import Searcher


class KeywordSearcher(Searcher):
    def __init__(self, search_config: SearchConfig):
        super().__init__(search_config)
        self.es_client = Elasticsearch(self.search_config.es_endpoint)

    def get_district_query(
        self,
        district_full_name: str,
        district_short_name: str,
        is_district_fuzzy: bool,
        boost_value: float = 1.0,
    ) -> Q:
        exact_district_query = (
            Q("match_phrase", Text={"query": district_full_name, "boost": boost_value})
            | Q(
                "match_phrase",
                Text={"query": district_short_name, "boost": boost_value},
            )
            | Q(
                "match_phrase",
                Text={
                    "query": district_short_name.replace("-", ""),
                    "boost": boost_value,
                },
            )
            | Q(
                "match_phrase",
                Text={
                    "query": district_short_name.replace(".", ""),
                    "boost": boost_value,
                },
            )
        )

        fuzzy_district_query = Q(
            "match", Text={"query": district_short_name, "fuzziness": "AUTO"}
        ) | Q("match", Text={"query": district_full_name, "fuzziness": "AUTO"})

        if is_district_fuzzy:
            district_query = Q(
                "bool", should=[exact_district_query, fuzzy_district_query]
            )
        else:
            district_query = exact_district_query

        return district_query

    def get_eval_term_query(
        self, eval_term: str, is_eval_term_fuzzy: bool, thesaurus_file: str
    ) -> Q:
        expanded_eval_term = self.expand_term(thesaurus_file, eval_term)
        exact_term_query = Q(
            "bool",
            should=list(Q("match_phrase", Text=t) for t in expanded_eval_term),
            minimum_should_match=1,
        )
        if is_eval_term_fuzzy:
            term_query = Q(
                "bool",
                should=[Q("match_phrase", Text=t) for t in expanded_eval_term]
                + [
                    Q("match", Text={"query": t, "fuzziness": "AUTO"})
                    for t in expanded_eval_term
                ],
                minimum_should_match=1,
            )
        else:
            term_query = exact_term_query

        return term_query

    def get_units_query(self, eval_term: str, thesaurus_file: str) -> Q:
        expanded_units = self.expand_term(thesaurus_file, f"{eval_term} units")
        units_query = Q(
            "bool",
            should=list(Q("match_phrase", Text=t) for t in expanded_units),
            minimum_should_match=1,
        )
        return units_query

    def search(self, search_query: SearchQuery, target: str) -> SearchResult | None:
        try:
            s = Search(using=self.es_client, index=search_query.place.town)
            is_district_fuzzy = self.search_config.is_district_fuzzy
            is_eval_term_fuzzy = self.search_config.is_eval_term_fuzzy

            # res = []
            # attempts = 0
            # max_attempts = 1

            # while len(res) == 0 and attempts < max_attempts:
            #     attempts += 1
            district_query = self.get_district_query(
                search_query.place.district_full_name,
                search_query.place.district_short_name,
                is_district_fuzzy,
            )
            eval_term_query = self.get_eval_term_query(
                search_query.eval_term,
                is_eval_term_fuzzy,
                self.search_config.thesaurus_file,
            )
            units_query = self.get_units_query(
                search_query.eval_term, self.search_config.thesaurus_file
            )

            s.query = district_query & eval_term_query & units_query

            s = s.extra(size=self.search_config.num_results)

            s = s.highlight("Text")

            res = s.execute()
            # if len(res) == 0:
            #     print(f"No results found for {target}")
            #     is_district_fuzzy = True
            #     is_eval_term_fuzzy = True

            if len(res) == 0:
                return SearchResult(
                    place=search_query.place,
                    eval_term=search_query.eval_term,
                    search_matches=[],
                )

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
            print(f"Error searching {target}")
            print(e)
            return None
