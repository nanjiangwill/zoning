from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from zoning.utils import flatten, page_coverage


class GlobalConfig(BaseModel):

    experiment_name: str
    experiment_dir: str
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
    page_embedding_dir: str
    district_extraction_dir: str
    district_extraction_verification_dir: str
    index_dir: str
    search_dir: str
    prompt_dir: str
    llm_dir: str
    normalization_dir: str
    eval_dir: str

    es_endpoint: str

    random_seed: int
    test_size_per_term: int

    config_file: str


class OCRConfig(BaseModel):

    method: str
    run_ocr: bool
    textract_region_name: str
    input_document_s3_bucket: str | None
    pdf_name_prefix_in_s3_bucket: str | None
    feature_types: List[str]


class FormatOCRConfig(BaseModel):
    temp: str


class DistrictExtractionConfig(BaseModel):
    run_district_extraction: bool
    embedding_model: str
    llm_model: str
    templates_dir: str
    system_prompt_file: str
    user_prompt_file: str
    verification_es_endpoint: str
    target_districts_file: str


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
    preprocess_search_target: bool


class PromptConfig(BaseModel):
    method: str
    max_search_pages: int
    templates_dir: str
    thesaurus_file: str


class LLMConfig(BaseModel):
    llm_name: str
    max_tokens: int
    formatted_response: bool
    cache_dir: str


class NormalizationConfig(BaseModel):
    method: str


class EvalConfig(BaseModel):
    template_dir: str


class ZoningConfig(BaseModel):
    config: Dict[str, Dict[str, str | int | bool | List[str] | None]]

    config_name: str = None
    global_config: GlobalConfig = None
    ocr_config: OCRConfig = None
    format_ocr_config: OCRConfig = None
    district_extraction_config: DistrictExtractionConfig = None
    index_config: IndexConfig = None
    search_config: SearchConfig = None
    prompt_config: PromptConfig = None
    llm_config: LLMConfig = None
    normalization_config: NormalizationConfig | None = None
    eval_config: EvalConfig | None = None

    def model_post_init(self, __context):
        self.config_name = self.config["global_config"]["experiment_name"]
        self.global_config = GlobalConfig(**self.config["global_config"])
        self.ocr_config = OCRConfig(**self.config["ocr_config"])
        self.format_ocr_config = FormatOCRConfig(**self.config["format_ocr_config"])
        self.district_extraction_config = DistrictExtractionConfig(
            **self.config["district_extraction_config"]
        )
        self.index_config = IndexConfig(**self.config["index_config"])
        self.search_config = SearchConfig(**self.config["search_config"])
        self.prompt_config = PromptConfig(**self.config["prompt_config"])
        self.llm_config = LLMConfig(**self.config["llm_config"])
        self.normalization_config = NormalizationConfig(
            **self.config["normalization_config"]
        )
        self.eval_config = EvalConfig(**self.config["eval_config"])


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

    def make_dict(self) -> Dict[str, str]:
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
        return {"page": str(self.page), "text": out}


class FormatOCR(BaseModel):
    """The formatted OCR representation for a town.

    Linked to format_ocr.
    """

    pages: List[Dict[str, str]]
    town: str


# =================
# Page Embedding and District Extraction
# =================
class PageEmbeddingResult(BaseModel):
    """Page embedding of a page in pdf.

    linked to district_extraction
    """

    town: str
    embedded_pages: List[Dict[str, str | List[float]]]


class DistrictExtractionResult(BaseModel):
    town: str
    districts: List[Dict[str, str]]
    districts_info_page: List[int]


class DistrictExtractionVerificationResult(BaseModel):
    town: str
    valid_districts: List[str]
    districts_info_page: List[int]


# =================
# Index
# =================


class ElasticSearchIndexData(BaseModel):
    index: str
    id: str
    document: Dict[str, str]
    request_timeout: int = 30


# =================
# Search
# =================


class Place(BaseModel):
    town: str
    district_short_name: str
    district_full_name: str

    def __str__(self) -> str:
        return f"{self.town}__{self.district_short_name}__{self.district_full_name}"

    def from_str(place_str):
        town, district_short_name, district_full_name = place_str.split("__")
        return Place(
            town=town,
            district_short_name=district_short_name,
            district_full_name=district_full_name,
        )


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
        if len(self.search_matches) > 0 and isinstance(self.search_matches[0], dict):
            self.search_matches = [SearchMatch(**d) for d in self.search_matches]
        self.entire_search_page_range = list(
            set(flatten(page_coverage([m.text for m in self.search_matches])))
        )
        self.entire_search_page_range.sort()


# =================
# Prompt
# =================


class Prompt(BaseModel):
    system_prompt: str
    user_prompt: str


class PromptResult(BaseModel):
    place: Place
    eval_term: str

    input_prompts: List[Prompt]

    def model_post_init(self, __context):
        if isinstance(self.place, dict):
            self.place = Place(**self.place)
        if len(self.input_prompts) > 0 and isinstance(self.input_prompts[0], dict):
            self.input_prompts = [Prompt(**d) for d in self.input_prompts]


# =================
# LLM Inference
# =================


class LLMOutput(BaseModel):
    place: Place
    eval_term: str
    raw_model_response: str | None = None
    extracted_text: Optional[List[str] | None] = None
    rationale: Optional[str | None] = None
    answer: Optional[str | None] = None

    def model_post_init(self, __context):
        if isinstance(self.place, dict):
            self.place = Place(**self.place)


class LLMInferenceResult(BaseModel):
    place: Place
    eval_term: str
    llm_outputs: List[LLMOutput]

    def model_post_init(self, __context):
        if isinstance(self.place, dict):
            self.place = Place(**self.place)
        if len(self.llm_outputs) > 0 and isinstance(self.llm_outputs[0], dict):
            self.llm_outputs = [LLMOutput(**d) for d in self.llm_outputs]


# =================
# Normalization
# =================


class NormalizedLLMOutput(BaseModel):
    llm_output: LLMOutput
    normalized_answer: List[str] | None


class NormalizedLLMInferenceResult(BaseModel):
    place: Place
    eval_term: str
    normalized_llm_outputs: List[NormalizedLLMOutput]

    def model_post_init(self, __context):
        if isinstance(self.place, dict):
            self.place = Place(**self.place)
        if len(self.normalized_llm_outputs) > 0 and isinstance(
            self.normalized_llm_outputs[0], dict
        ):
            self.normalized_llm_outputs = [
                NormalizedLLMOutput(**d) for d in self.normalized_llm_outputs
            ]


# =================
# Eval
# =================


class EvalResult(BaseModel):
    place: Place
    eval_term: str

    ground_truth: str | None
    ground_truth_orig: str | None
    ground_truth_page: str | None
    answer_correct: bool | None
    page_in_range: bool | None

    def model_post_init(self, __context):
        if isinstance(self.place, dict):
            self.place = Place(**self.place)
