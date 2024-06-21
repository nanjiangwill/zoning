from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from zoning.utils import flatten, page_coverage


class GlobalConfig(BaseModel):

    experiment_name: str
    target_state: str
    eval_terms: List[str]

    target_town_file: str
    target_district_file: str
    target_eval_file: str
    ground_truth_file: str
    thesaurus_file: str

    pdf_dir: str
    ocr_dir: str
    format_ocr_dir: str
    index_dir: str
    search_dir: str
    llm_dir: str
    normalization_dir: str
    eval_dir: str

    es_endpoint: str

    random_seed: int
    test_size_per_term: int


class OCRConfig(BaseModel):

    method: str
    run_ocr: bool
    input_document_s3_bucket: str | None
    pdf_name_prefix_in_s3_bucket: str | None
    feature_types: List[str]


class FormatOCRConfig(BaseModel):
    temp: str


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
    template_dir: str


class ZoningConfig(BaseModel):
    config: Dict[str, Dict[str, str | int | bool | List[str] | None]]

    global_config: GlobalConfig = None
    ocr_config: OCRConfig = None
    format_ocr_config: OCRConfig = None
    index_config: IndexConfig = None
    search_config: SearchConfig = None
    llm_config: LLMConfig = None
    normalization_config: NormalizationConfig | None = None
    eval_config: EvalConfig | None = None

    def model_post_init(self, __context):
        self.global_config = GlobalConfig(**self.config["global_config"])
        self.ocr_config = OCRConfig(**self.config["ocr_config"])
        self.format_ocr_config = FormatOCRConfig(**self.config["format_ocr_config"])
        self.index_config = IndexConfig(**self.config["index_config"])
        self.search_config = SearchConfig(**self.config["search_config"])
        self.llm_config = LLMConfig(**self.config["llm_config"])
        self.normalization_config = NormalizationConfig(
            **self.config["normalization_config"]
        )
        self.eval_config = EvalConfig(**self.config["eval_config"])


# class OCREntity(BaseModel):
#     name: str
#     pdf_file: str
#     ocr_results_file: str


# class OCREntities(BaseModel):
#     target_names_file: str
#     pdf_dir: str
#     ocr_results_dir: str
#     ocr_entities: List[OCREntity] = []

#     def model_post_init(self, __context):
#         target_names = json.load(open(self.target_names_file))
#         self.ocr_entities = [
#             OCREntity(
#                 name=name,
#                 pdf_file=os.path.join(self.pdf_dir, f"{name}.pdf"),
#                 ocr_results_file=os.path.join(self.ocr_results_dir, f"{name}.json"),
#             )
#             for name in target_names
#         ]
#         os.makedirs(self.pdf_dir, exist_ok=True)
#         os.makedirs(self.ocr_results_dir, exist_ok=True)


# =================
# Format OCR
# =================


class OCRBlock(BaseModel):
    id: str
    text: str
    typ: str
    relationships: List[str]
    position: Tuple[int, int]


class OCRPage(BaseModel):
    ents: List[OCRBlock] = []
    seen: Dict[str, bool] = {}
    relations: Dict[str, List[OCRBlock]] = {}
    page: int = 0

    def add(self, entity: OCRBlock):
        if entity.id in self.seen:
            return
        self.ents.append(entity)
        self.seen[entity.id] = True
        for r in entity.relationships:
            self.relations.setdefault(r, [])
            self.relations[r].append(entity)

    def make_string(self) -> str:
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


class FormatOCR(BaseModel):
    """The formatted OCR representation for a town.

    Linked to format_ocr.
    """

    pages: List[str]
    town_name: str


# ==================


# =================
# Index
# =================


class ElasticSearchIndexData(BaseModel):
    index: str
    id: str
    document: Dict[str, str]
    request_timeout: int = 30


# class IndexEntity(BaseModel):
#     name: str
#     page_data: List[Dict[str, str | int]]


# class IndexEntities(BaseModel):
#     index_entities: List[IndexEntity]


# =================
# Search
# =================


class Place(BaseModel):
    town: str
    district_short_name: str
    district_full_name: str

    def __str__(self) -> str:
        return f"{self.town}__{self.district_short_name}__{self.district_full_name}"


class SearchQuery(BaseModel):
    raw_query_str: str

    place: Place = None
    eval_term: str = ""

    def model_post_init(self, __context):
        eval_term, town, district_short_name, district_full_name = (
            self.raw_query_str.split("__")
        )
        self.place = Place(
            town=town,
            district_short_name=district_short_name,
            district_full_name=district_full_name,
        )
        self.eval_term = eval_term

    def __str__(self) -> str:
        return f"{self.eval_term}__{self.place}"


# class SearchQueries(BaseModel):
#     query_file: str

#     search_queries: List[SearchQuery] = []
#     search_queries_by_eval_term: Dict[str, List[SearchQuery]] = {}

#     def model_post_init(self, __context):
#         query_data = json.load(open(self.query_file))

#         all_eval_terms = [
#             i.replace("_page_gt", "")
#             for i in query_data[0].keys()
#             if i.endswith("_page_gt")
#         ]

#         self.search_queries = [
#             SearchQuery(
#                 place=Place(
#                     town=d["town"],
#                     district_full_name=d["district"],
#                     district_short_name=d["district_abb"],
#                 ),
#                 eval_term=eval_term,
#             )
#             for d in query_data
#             for eval_term in all_eval_terms
#         ]
#         self.search_queries_by_eval_term = {
#             eval_term: [q for q in self.search_queries if q.eval_term == eval_term]
#             for eval_term in all_eval_terms
#         }

#     def get_test_data_search_queries(
#         self, eval_terms: List[str], random_seed: int, test_size_per_term: int
#     ) -> None:
#         random.seed(random_seed)
#         test_data_search_queries = []
#         for eval_term in eval_terms:
#             search_queries = self.search_queries_by_eval_term[eval_term]
#             test_data_search_queries += random.sample(
#                 search_queries, test_size_per_term
#             )
#         self.search_queries = test_data_search_queries
#         self.search_queries_by_eval_term = {
#             eval_term: [q for q in self.search_queries if q.eval_term == eval_term]
#             for eval_term in set([q.eval_term for q in self.search_queries])
#         }


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
    entire_search_page_range: List[int] = []

    def model_post_init(self, __context):
        if isinstance(self.place, dict):
            self.place = Place(**self.place)
        if isinstance(self.search_matches[0], dict):
            self.search_matches = [SearchMatch(**d) for d in self.search_matches]
        self.entire_search_page_range = list(
            set(flatten(page_coverage([m.text for m in self.search_matches])))
        )
        self.entire_search_page_range.sort()


# class SearchResults(BaseModel):
#     search_results: List[SearchResult]

#     def model_post_init(self, __context):
#         if isinstance(type(self.search_results[0]), dict):
#             self.search_results = [SearchResult(**d) for d in self.search_results]


# =================
# LLM Inference
# =================


class LLMQuery(BaseModel):
    place: Place
    eval_term: str
    context: str


class LLMOutput(BaseModel):
    place: Place
    eval_term: str
    search_match: SearchMatch | None
    input_prompt: List[Dict[str, str]] | str
    search_page_range: List[int] | None = []
    raw_model_response: str | None = None
    extracted_text: Optional[List[str] | None] = None
    rationale: Optional[str | None] = None
    answer: Optional[str | None] = None

    def model_post_init(self, __context):
        self.search_page_range = sorted(
            flatten(page_coverage([self.search_match.text]))
        )


class LLMInferenceResult(BaseModel):
    place: Place
    eval_term: str
    search_result: SearchResult
    llm_outputs: List[LLMOutput]

    def model_post_init(self, __context):
        if isinstance(self.llm_outputs[0], dict):
            self.llm_outputs = [LLMOutput(**d) for d in self.llm_outputs]


# class LLMInferenceResults(BaseModel):
#     llm_inference_results: List[LLMInferenceResult]
#     llm_inference_results_by_eval_term: Dict[str, List[LLMInferenceResult]] = {}

#     def model_post_init(self, __context):
#         if isinstance(self.llm_inference_results[0], dict):
#             self.llm_inference_results = [
#                 LLMInferenceResult(**d) for d in self.llm_inference_results
#             ]
#         self.llm_inference_results_by_eval_term = {
#             eval_term: [
#                 r for r in self.llm_inference_results if r.eval_term == eval_term
#             ]
#             for eval_term in set([r.eval_term for r in self.llm_inference_results])
#         }


# =================
# Normalization
# =================


class NormalizedLLMOutput(BaseModel):
    llm_output: LLMOutput
    normalized_answer: List[str] | None


class NormalizedLLMInferenceResult(BaseModel):
    place: Place
    eval_term: str
    search_result: SearchResult
    normalized_llm_outputs: List[NormalizedLLMOutput]

    def model_post_init(self, __context):
        if isinstance(self.place, dict):
            self.place = Place(**self.place)

        if isinstance(self.search_result, dict):
            self.search_result = SearchResult(**self.search_result)
        if isinstance(self.normalized_llm_outputs[0], dict):
            self.normalized_llm_outputs = [
                NormalizedLLMOutput(**d) for d in self.normalized_llm_outputs
            ]


# =================
# Eval
# =================


class DistrictEvalResult(BaseModel):
    place: Place
    eval_term: str
    search_result: SearchResult
    normalized_llm_outputs: List[NormalizedLLMOutput]
    ground_truth: str | None
    ground_truth_orig: str | None
    ground_truth_page: str | None
    answer_correct: bool | None
    page_in_range: bool | None

    def model_post_init(self, __context):
        if isinstance(self.place, dict):
            self.place = Place(**self.place)
        if isinstance(self.search_result, dict):
            self.search_result = SearchResult(**self.search_result)
        if isinstance(self.normalized_llm_outputs[0], dict):
            self.normalized_llm_outputs = [
                NormalizedLLMOutput(**d) for d in self.normalized_llm_outputs
            ]


# class EvalQuery(BaseModel):
#     place: Place
#     eval_term: str
#     search_result: SearchResult
#     llm_inference_result: LLMInferenceResult
#     ground_truth: str | None
#     ground_truth_orig: str | None
#     ground_truth_page: str | None


# class EvalQueries(BaseModel):
#     eval_queries: List[EvalQuery]

#     def model_post_init(self, __context):
#         if isinstance(self.eval_queries[0], dict):
#             self.eval_queries = [EvalQuery(**d) for d in self.eval_queries]


# class EvalMetricByTerm(BaseModel):
#     eval_term: str
#     answer_accuracy: float
#     page_precision: float
