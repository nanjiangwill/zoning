import json
import os
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from zoning.utils import flatten, page_coverage


class GlobalConfig(BaseModel):

    experiment_name: str
    target_state: str
    eval_terms: List[str]

    target_names_file: str
    ground_truth_file: str
    thesaurus_file: str

    pdf_dir: str
    ocr_results_dir: str

    es_endpoint: str

    data_flow_ocr_file: str
    data_flow_index_file: str
    data_flow_search_file: str
    data_flow_llm_file: str
    data_flow_eval_file: str


class OCRConfig(BaseModel):

    method: str
    run_ocr: bool
    input_document_s3_bucket: str | None
    pdf_name_prefix_in_s3_bucket: str | None
    feature_types: List[str]


class IndexConfig(BaseModel):
    method: str
    index_key: str
    es_endpoint: str
    index_range: int


class SearchConfig(BaseModel):
    method: str
    es_endpoint: str
    num_results: int
    is_district_fuzzy: bool
    is_eval_term_fuzzy: bool
    thesaurus_file: str


class LLMConfig(BaseModel):

    method: str
    llm_name: str
    max_tokens: int
    templates_dir: str
    formatted_response: bool
    cache_dir: str
    thesaurus_file: str


class NormalizationConfig(BaseModel):

    method: str


class EvalConfig(BaseModel):
    random_seed: int
    test_size_per_term: int


class ZoningConfig(BaseModel):
    config: Dict[str, Dict[str, str | int | bool | List[str] | None]]

    global_config: GlobalConfig = None
    ocr_config: OCRConfig = None
    index_config: IndexConfig = None
    search_config: SearchConfig = None
    llm_config: LLMConfig = None
    normalization_config: NormalizationConfig | None = None
    eval_config: EvalConfig = None

    def model_post_init(self, __context):
        self.global_config = GlobalConfig(**self.config["global_config"])
        self.ocr_config = OCRConfig(**self.config["ocr_config"])
        self.index_config = IndexConfig(**self.config["index_config"])
        self.search_config = SearchConfig(**self.config["search_config"])
        self.llm_config = LLMConfig(**self.config["llm_config"])
        self.normalization_config = NormalizationConfig(
            **self.config["normalization_config"]
        )
        self.eval_config = EvalConfig(**self.config["eval_config"])


class OCREntity(BaseModel):
    name: str
    pdf_file: str
    ocr_results_file: str


class OCREntities(BaseModel):
    target_names_file: str
    pdf_dir: str
    ocr_results_dir: str
    ocr_entities: List[OCREntity] = []

    def model_post_init(self, __context):
        with open(self.target_names_file, "r") as f:
            target_names = json.load(f)
        self.ocr_entities = [
            OCREntity(
                name=name,
                pdf_file=os.path.join(self.pdf_dir, f"{name}.pdf"),
                ocr_results_file=os.path.join(self.ocr_results_dir, f"{name}.json"),
            )
            for name in target_names
        ]
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.ocr_results_dir, exist_ok=True)


class Place(BaseModel):
    town: str
    district_full_name: str
    district_short_name: str

    def __str__(self) -> str:
        return f"{self.town}-{self.district_full_name}"


class OCRPageResult(BaseModel):
    id: str
    text: str
    typ: str
    relationships: List[str]
    position: Tuple[int, int]


class OCRPageResults(BaseModel):
    ents: List[OCRPageResult]
    seen: Set[str]
    relations: Dict[str, List[OCRPageResult]]

    def add(self, entity: OCRPageResult):
        if entity.id in self.seen:
            return
        self.ents.append(entity)
        self.seen.add(entity.id)
        for r in entity.relationships:
            self.relations.setdefault(r, [])
            self.relations[r].append(entity)

    def __str__(self) -> str:
        out = ""
        for e in self.ents:
            if e.typ == "LINE":
                in_cell = [
                    o
                    for r in e.relationships
                    for o in self.relations[r]
                    if o.typ == "CELL"
                ]
                if not in_cell:
                    out += e.text + "\n"
            if e.typ == "CELL":
                lines = [
                    o
                    for r in e.relationships
                    for o in self.relations[r]
                    if o.typ == "LINE"
                ]

                out += f"CELL {e.position}: \n"
                seen = set()
                for o in lines:
                    if o.id in seen:
                        continue
                    seen.add(o.id)
                    out += o.text + "\n"
        return out


class ElasticSearchIndexData(BaseModel):
    index: str
    id: str
    document: Dict[str, str]
    request_timeout: int = 30


class IndexEntity(BaseModel):
    name: str
    page_data: List[Dict[str, str | int]]


class IndexEntities(BaseModel):
    index_entities: List[IndexEntity]


class SearchQuery(BaseModel):
    place: Place
    eval_term: str

    def get_index_key(self) -> str:
        return self.place.town


class SearchQueries(BaseModel):
    search_queries: List[SearchQuery]


class SearchMatch(BaseModel):
    text: str
    page_number: int
    page_range: List[int] = []
    highlight: List[str]
    score: float
    query: str

    def model_post_init(self, __context):
        self.page_range = flatten(page_coverage([self.text]))


class SearchResult(BaseModel):
    place: Place
    eval_term: str
    search_matches: List[SearchMatch]
    entire_search_page_range: Set[int]

    def model_post_init(self, __context):
        self.entire_search_page_range = flatten(
            page_coverage([m.text for m in self.search_matches])
        )


class SearchResults(BaseModel):
    search_results: List[SearchResult]

    def model_post_init(self, __context):
        self.search_results = [SearchResult(**d) for d in self.search_results]


class LLMQuery(BaseModel):
    place: Place
    eval_term: str
    search_match: SearchMatch


class LLMOutput(BaseModel):
    place: Place
    eval_term: str
    search_match: SearchMatch
    input_prompt: List[Dict[str, str]] | str
    search_page_range: Set[int]
    raw_model_response: str | None = None
    extracted_text: Optional[List[str] | None] = None
    rationale: Optional[str | None] = None
    answer: Optional[str | None] = None


class LLMInferenceResult(BaseModel):
    place: Place
    eval_term: str
    search_result: SearchResult
    llm_inference_result: List[LLMOutput]


class LLMInferenceResults(BaseModel):
    llm_inference_results: List[LLMInferenceResult]
    llm_inference_results_by_eval_term: Dict[str, List[LLMInferenceResult]] = {}

    def model_post_init(self, __context):
        self.llm_inference_results_by_eval_term = {
            eval_term: [
                r for r in self.llm_inference_results if r.eval_term == eval_term
            ]
            for eval_term in set([r.eval_term for r in self.llm_inference_results])
        }


class EvalQuery(BaseModel):
    place: Place
    eval_term: str
    search_result: SearchResult
    llm_inference_result: LLMInferenceResult
    ground_truth: str
    ground_truth_orig: str
    ground_truth_page: str


class EvalQueries(BaseModel):
    evaluation_queries: List[EvalQuery]

    def model_post_init(self, __context):
        self.evaluation_queries = [EvalQuery(**d) for d in self.evaluation_queries]


class EvalMetricByTerm(BaseModel):
    eval_term: str
    answer_f1: float
    answer_precision: float
    answer_recall: float
    page_f1: float
    page_precision: float
    page_recall: float
    is_in_entire_search_page_range: bool
