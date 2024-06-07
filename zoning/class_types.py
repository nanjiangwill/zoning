from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json
from elasticsearch_dsl import Q
from pydantic import BaseModel
from utils import expand_term, flatten, page_coverage


@dataclass
class Place(BaseModel):
    town: str
    district_full_name: str
    district_short_name: str


@dataclass
class SearchEvalTermPattern(BaseModel):
    name: str
    is_eval_term_fuzzy: bool
    thesaurus_file: str
    expanded_eval_term: Q
    expanded_units: Q

    def __post_init__(self):
        self.expanded_eval_term = expand_term(self.thesaurus_file, self.name)
        self.expanded_units = expand_term(self.thesaurus_file, f"{self.name} units")

    def get_term_query(self) -> Q:
        exact_term_query = Q(
            "bool",
            should=list(Q("match_phrase", Text=t) for t in self.expanded_term),
            minimum_should_match=1,
        )
        if self.is_eval_term_fuzzy:
            term_query = Q(
                "bool",
                should=[Q("match_phrase", Text=t) for t in self.expanded_term]
                + [
                    Q("match", Text={"query": t, "fuzziness": "AUTO"})
                    for t in self.expanded_term
                ],
                minimum_should_match=1,
            )
        else:
            term_query = exact_term_query

        return term_query

    def get_units_query(self) -> Q:
        units_query = Q(
            "bool",
            should=list(Q("match_phrase", Text=t) for t in self.expanded_units),
            minimum_should_match=1,
        )
        return units_query

    def get_query(self) -> Q:
        return self.get_term_query() & self.get_units_query()

    def __str__(self) -> str:
        return self.name


@dataclass
class SearchPlacePattern(BaseModel):
    place: Place
    is_district_fuzzy: bool

    def get_district_query(self) -> Q:
        # Boost factor: Increasing the boost value will make documents matching this query to be ranked higher
        # Reference to Fuzzy: https://blog.mikemccandless.com/2011/03/lucenes-fuzzyquery-is-100-times-faster.html
        boost_value = 1.0
        exact_district_query = (
            Q(
                "match_phrase",
                Text={"query": self.place.district_full_name, "boost": boost_value},
            )
            | Q(
                "match_phrase",
                Text={"query": self.place.district_short_name, "boost": boost_value},
            )
            | Q(
                "match_phrase",
                Text={
                    "query": self.place.district_short_name.replace("-", ""),
                    "boost": boost_value,
                },
            )
            | Q(
                "match_phrase",
                Text={
                    "query": self.place.district_short_name.replace(".", ""),
                    "boost": boost_value,
                },
            )
        )

        fuzzy_district_query = Q(
            "match", Text={"query": self.place.district_short_name, "fuzziness": "AUTO"}
        ) | Q(
            "match", Text={"query": self.place.district_full_name, "fuzziness": "AUTO"}
        )

        if self.is_district_fuzzy:
            district_query = Q(
                "bool", should=[exact_district_query, fuzzy_district_query]
            )
        else:
            district_query = exact_district_query

        return district_query

    def get_query(self) -> Q:
        return self.get_district_query()


@dataclass
class SearchPattern(BaseModel):
    search_eval_term_pattern: SearchEvalTermPattern
    search_place_pattern: SearchPlacePattern

    def get_query(self) -> Q:
        return (
            self.search_eval_term_pattern.get_query()
            & self.search_place_pattern.get_query()
        )

    def get_index_key(self) -> str:
        return self.search_place_pattern.place.town

    def __str__(self) -> str:
        return f"{self.search_eval_term_pattern.name} in {self.search_place_pattern.place.town} {self.search_place_pattern.place.district_full_name}"


@dataclass
class EvaluationData(BaseModel):
    place: Place
    eval_term: str
    is_eval_term_fuzzy: bool
    is_district_fuzzy: bool
    thesaurus_file: str
    search_pattern: SearchPattern

    def __post_init__(self):
        self.search_pattern = SearchPattern(
            SearchEvalTermPattern(
                name=self.eval_term,
                is_eval_term_fuzzy=self.is_eval_term_fuzzy,
                thesaurus_file=self.thesaurus_file,
            ),
            SearchPlacePattern(
                place=self.place, is_district_fuzzy=self.is_district_fuzzy
            ),
        )


@dataclass_json
@dataclass
class SearchResult(BaseModel):
    text: str
    page_number: int
    page_range: list[int]
    highlight: list[str]
    score: float
    query: str

    def __post_init__(self):
        self.page_range = flatten(page_coverage([self.text]))


@dataclass
class LLMQuery(BaseModel):
    place: Place
    eval_term: str
    search_results: list[SearchResult]


@dataclass_json
@dataclass
class LLMInferenceResult(BaseModel):
    search_result: SearchResult

    extracted_text: list[str]  # Check
    rationale: str
    answer: Optional[str | None]


@dataclass
class EvaluationDataResults(BaseModel):
    place: Place
    eval_term: str
    entire_searched_page_range: list[int]
    search_results: list[SearchResult]
    llm_inference_results: list[LLMInferenceResult]

    def __post_init__(self):
        self.search_results = [r.search_result for r in self.llm_inference_results]
