import json
import os
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from zoning.utils import flatten, page_coverage


class Place(BaseModel):
    town: str
    district_full_name: str
    district_short_name: str

    def __str__(self) -> str:
        return f"{self.town}-{self.district_full_name}"


class ExtractionEntity(BaseModel):
    name: str
    pdf_file: str
    ocr_result_file: str
    dataset_file: str


class ExtractionEntities(BaseModel):
    pdf_dir: str
    ocr_result_dir: str
    dataset_dir: str
    target_names_file: str
    targets: List[ExtractionEntity] = []

    def model_post_init(self, __context):
        with open(self.target_names_file, "r") as f:
            target_names = json.load(f)
        self.targets = [
            ExtractionEntity(
                name=name,
                pdf_file=os.path.join(self.pdf_dir, f"{name}.pdf"),
                ocr_result_file=os.path.join(self.ocr_result_dir, f"{name}.json"),
                dataset_file=os.path.join(self.dataset_dir, f"{name}.json"),
            )
            for name in target_names
        ]
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.ocr_result_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)


class ExtractionResult(BaseModel):
    id: str
    text: str
    typ: str
    relationships: List[str]
    position: Tuple[int, int]


class ExtractionResults(BaseModel):
    ents: List[ExtractionResult]
    seen: Set[str]
    relations: Dict[str, List[ExtractionResult]]

    def add(self, entity: ExtractionResult):
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
    dataset_dir: str
    index_range: int
    dataset_file: str = ""

    def model_post_init(self, __context):
        self.dataset_file = os.path.join(self.dataset_dir, f"{self.name}.json")


class IndexEntities(BaseModel):
    dataset_dir: str
    index_range: int
    target_names_file: str
    index_entities: List[IndexEntity] = []

    def model_post_init(self, __context):
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset directory {self.dataset_dir} does not exist"
        assert os.listdir(
            self.dataset_dir
        ), f"Dataset directory {self.dataset_dir} is empty"

        with open(self.target_names_file, "r") as f:
            target_names = json.load(f)
        # simple detect if there are any files that are not extracted
        # can be deleted later
        missing_files = [
            name
            for name in target_names
            if not os.path.exists(os.path.join(self.dataset_dir, f"{name}.json"))
        ]
        print("Missing files: ", missing_files)
        print("Total missing files: ", len(missing_files))
        print("Total files: ", len(target_names))

        self.index_entities = [
            IndexEntity(
                name=name, dataset_dir=self.dataset_dir, index_range=self.index_range
            )
            for name in target_names
            if os.path.exists(os.path.join(self.dataset_dir, f"{name}.json"))
        ]


class SearchResult(BaseModel):
    text: str
    page_number: int
    page_range: List[int] = []
    highlight: List[str]
    score: float
    query: str

    def model_post_init(self, __context):
        self.page_range = flatten(page_coverage([self.text]))


class LLMQuery(BaseModel):
    place: Place
    eval_term: str
    context: str


class LLMQueries(BaseModel):
    place: Place
    eval_term: str
    search_results: List[SearchResult]
    query_list: List[LLMQuery] = []

    def model_post_init(self, __context):
        self.query_list = [
            LLMQuery(
                place=self.place, eval_term=self.eval_term, context=search_result.text
            )
            for search_result in self.search_results
        ]


class LLMInferenceResult(BaseModel):
    input_prompt: List[Dict[str, str]] | str
    search_page_range: Set[int]
    raw_model_response: str | None = None
    extracted_text: Optional[List[str] | None] = None
    rationale: Optional[str | None] = None
    answer: Optional[str | None] = None


class EvaluationDatum(BaseModel):
    place: Place
    eval_term: str
    is_eval_term_fuzzy: bool
    is_district_fuzzy: bool
    thesaurus_file: str

    def get_index_key(self) -> str:
        return self.place.town

    def __str__(self) -> str:
        return f"{self.eval_term} in {self.place}"


class EvaluationDatumResult(BaseModel):
    place: Place
    eval_term: str
    search_results: List[SearchResult]
    llm_inference_results: List[LLMInferenceResult]
    entire_search_results_page_range: set[int] = ()
    ground_truth: str | None = None
    ground_truth_orig: str | None = None
    ground_truth_page: str | None = None

    def model_post_init(self, __context):
        self.entire_search_results_page_range = set(
            [i for r in self.search_results for i in r.page_range]
        )


class AllEvaluationResults(BaseModel):
    all_evaluation_results: List[EvaluationDatumResult]

    all_evaluation_results_by_town: Dict[str, List[EvaluationDatumResult]] = {}
    all_evaluation_results_by_district: Dict[str, List[EvaluationDatumResult]] = {}
    all_evaluation_results_by_eval_term: Dict[str, List[EvaluationDatumResult]] = {}

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
