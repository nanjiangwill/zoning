import json
import os
from typing import Iterable, Optional

from elasticsearch_dsl import Q
from pydantic import BaseModel
from utils import expand_term, flatten, page_coverage


class Place(BaseModel):
    town: str
    district_full_name: str
    district_short_name: str

    def __str__(self) -> str:
        return f"{self.town}-{self.district_full_name}"


class SearchEvalTermPattern(BaseModel):
    name: str
    is_eval_term_fuzzy: bool
    thesaurus_file: str
    expanded_eval_term: Iterable[str] = []
    expanded_units: Iterable[str] = []

    def model_post_init(self, __context):
        self.expanded_eval_term = expand_term(self.thesaurus_file, self.name)
        self.expanded_units = expand_term(self.thesaurus_file, f"{self.name} units")

    def get_term_query(self) -> Q:
        exact_term_query = Q(
            "bool",
            should=list(Q("match_phrase", Text=t) for t in self.expanded_eval_term),
            minimum_should_match=1,
        )
        if self.is_eval_term_fuzzy:
            term_query = Q(
                "bool",
                should=[Q("match_phrase", Text=t) for t in self.expanded_eval_term]
                + [
                    Q("match", Text={"query": t, "fuzziness": "AUTO"})
                    for t in self.expanded_eval_term
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


class SearchPattern(BaseModel):
    search_eval_term_pattern: SearchEvalTermPattern
    search_place_pattern: SearchPlacePattern

    def get_place(self) -> Place:
        return self.search_place_pattern.place

    def get_eval_term(self) -> str:
        return self.search_eval_term_pattern.name

    def get_query(self) -> Q:
        return (
            self.search_eval_term_pattern.get_query()
            & self.search_place_pattern.get_query()
        )

    def get_index_key(self) -> str:
        return self.search_place_pattern.place.town

    def __str__(self) -> str:
        return (
            f"{self.search_eval_term_pattern.name} in {self.search_place_pattern.place}"
        )


class EvaluationDatum(BaseModel):
    place: Place
    eval_term: str
    is_eval_term_fuzzy: bool
    is_district_fuzzy: bool
    thesaurus_file: str
    search_pattern: SearchPattern = None

    def model_post_init(self, __context):
        self.search_pattern = SearchPattern(
            search_eval_term_pattern=SearchEvalTermPattern(
                name=self.eval_term,
                is_eval_term_fuzzy=self.is_eval_term_fuzzy,
                thesaurus_file=self.thesaurus_file,
            ),
            search_place_pattern=SearchPlacePattern(
                place=self.place, is_district_fuzzy=self.is_district_fuzzy
            ),
        )


class SearchResult(BaseModel):
    text: str
    page_number: int
    page_range: list[int] = []
    highlight: list[str]
    score: float
    query: str

    def model_post_init(self, __context):
        self.page_range = flatten(page_coverage([self.text]))

    # def to_json():
    #     # TODO with @dataclass_json
    #     pass


class LLMQuery(BaseModel):
    place: Place
    eval_term: str
    context: str


class LLMQueries(BaseModel):
    place: Place
    eval_term: str
    search_results: list[SearchResult]
    query_list: list[LLMQuery] = []

    def model_post_init(self, __context):
        self.query_list = [
            LLMQuery(
                place=self.place, eval_term=self.eval_term, context=search_result.text
            )
            for search_result in self.search_results
        ]


class LLMInferenceResult(BaseModel):
    input_prompt: list[dict[str, str]]
    raw_model_response: str | None = None
    extracted_text: Optional[list[str] | None] = None
    rationale: Optional[str | None] = None
    answer: Optional[str | None] = None


class EvaluationDatumResult(BaseModel):
    place: Place
    eval_term: str
    search_results: list[SearchResult]
    llm_inference_results: list[LLMInferenceResult]
    entire_search_results_page_range: list[int] = []
    ground_truth: str | None = None
    ground_truth_page: str | None = None

    def model_post_init(self, __context):
        self.entire_search_results_page_range = set(
            [i for r in self.search_results for i in r.page_range]
        )


class AllEvaluationResults(BaseModel):
    all_evaluation_results: list[EvaluationDatumResult]

    all_evaluation_results_by_town: dict[str, list[EvaluationDatumResult]] = {}
    all_evaluation_results_by_district: dict[str, list[EvaluationDatumResult]] = {}
    all_evaluation_results_by_eval_term: dict[str, list[EvaluationDatumResult]] = {}

    def model_post_init(self, __context):
        for evaluation_result in self.all_evaluation_results:
            self.all_evaluation_results_by_town.setdefault(
                evaluation_result.place.town, []
            ).append(evaluation_result)
            self.all_evaluation_results_by_district.setdefault(
                evaluation_result.place.district_full_name, []
            ).append(evaluation_result)
            self.all_evaluation_results_by_eval_term.setdefault(
                evaluation_result.eval_term, []
            ).append(evaluation_result)

    def save_to(self, result_output_dir: str):
        # we can name experiment
        os.makedirs(
            os.path.join(result_output_dir),
            exist_ok=True,
        )
        for (
            eval_term,
            eval_term_results,
        ) in self.all_evaluation_results_by_eval_term.items():
            eval_term_output_dir = os.path.join(
                result_output_dir,
                f"{eval_term}.json",
            )
            data = []
            for evaluation_result in eval_term_results:
                data.append(evaluation_result.model_dump_json())
            with open(eval_term_output_dir, "w") as f:
                json.dump(data, f)


class EvaluationMetricByTerm(BaseModel):
    eval_term: str
    answer_f1: float
    answer_precision: float
    answer_recall: float
    page_f1: float
    page_precision: float
    page_recall: float
    is_in_entire_search_page_range: bool
