import asyncio
import inspect
import json
import re
from functools import partial, wraps
from typing import Any, Iterable, TypeVar

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from openai import APIConnectionError, APIError, OpenAI, RateLimitError, Timeout
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, wait_random_exponential
from typer import Typer

T = TypeVar("T")


class District(BaseModel):
    full_name: str
    short_name: str


class PageSearchOutput(BaseModel):
    text: str
    page_number: int
    highlight: list[str]
    score: float
    query: str


class ExtractionOutput(BaseModel):
    extracted_text: list[str]
    rationale: str
    answer: str | None

    def __str__(self):
        return f"ExtractionOutput(extracted_text={self.extracted_text}, rationale={self.rationale}, answer={self.answer})"


class ExtractionOutput2(BaseModel):
    district_explanation: str
    district: str
    term_explanation: str
    term: str
    explanation: str
    answer: str | None

    def __str__(self):
        return f"ExtractionOutput(extracted_text={self.extracted_text}, rationale={self.rationale}, answer={self.answer})"


class LookupOutput(BaseModel):
    output: ExtractionOutput | ExtractionOutput2 | None
    search_pages: list[PageSearchOutput]
    search_pages_expanded: list[int]
    """The set of pages, in descending order or relevance, used to produce the
    result."""

    def __str__(self):
        return f"LookupOutput(output={self.output}, search_pages=[...], search_pages_expanded={self.search_pages_expanded})"

    def to_dict(self):
        return json.loads(self.model_dump_json())


class EvaluationMetrics(BaseModel):
    town: str
    district: str
    term: str
    gt_page: list[int]
    correct_page_searched: bool
    this_correct_page_searched: bool
    expanded_pages: list[int] | None
    extracted_pages: list[int] | None
    extracted_pages_expanded: list[int] | None
    expected: int | float
    expected_extended: Any
    pages: list[int] | None
    rationale: str | None
    extracted_text: str | None
    actual: str | None
    confirmed_flag: None
    confirmed_raw: None
    actual_before_confirmation: None


# Copied from https://github.com/tiangolo/typer/issues/88
class AsyncTyper(Typer):
    @staticmethod
    def maybe_run_async(decorator, f):
        if inspect.iscoroutinefunction(f):

            @wraps(f)
            def runner(*args, **kwargs):
                return asyncio.run(f(*args, **kwargs))

            decorator(runner)
        else:
            decorator(f)
        return f

    def callback(self, *args, **kwargs):
        decorator = super().callback(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)

    def command(self, *args, **kwargs):
        decorator = super().command(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)


def normalize_town(x) -> str:
    x = x.lower().strip()
    x = re.sub(r"\s*-\s*", "-", x)
    x = re.sub(r"\s*/\s*", "-", x)
    x = re.sub(r"\s+", "-", x)
    return x


def get_thesaurus(thesarus_file) -> dict:
    with open(thesarus_file, "r", encoding="utf-8") as f:
        thesaurus = json.load(f)
    return thesaurus


def expand_term(thesarus_file: str, term: str) -> Iterable[str]:
    # term = term.replace("_", " ").strip()
    # logger.info(f"Term: {term}")  # Initial logging of the term
    thesarus = get_thesaurus(thesarus_file)
    min_variations = thesarus.get("min", [])
    max_variations = thesarus.get("max", [])
    expanded_count = 0
    for query in thesarus.get(term, []):  # Iterate over thesaurus entries for the term
        # query = query.replace("_", " ").strip()
        if "min" in query or "minimum" in query:  # Handling minimum variations
            for r in min_variations:
                modified_query = query.replace(
                    "min", r
                )  # Replace 'min' with its variations
                # logger.info(f"Yielding: {modified_query}")  # Log the value to be yielded
                expanded_count += 1
                yield modified_query
        elif "max" in query or "maximum" in query:  # Handling maximum variations
            for r in max_variations:
                modified_query = query.replace(
                    "max", r
                )  # Replace 'max' with its variations
                # logger.info(f"Yielding: {modified_query}")  # Log the value to be yielded
                expanded_count += 1
                yield modified_query
        else:
            # logger.info(f"Yielding: {query}")  # Log the unmodified query to be yielded
            expanded_count += 1
            yield query
    # logger.info(f"Expanded {term} to {expanded_count} variations.")  # Log the total number of variations


def get_town_district_mapping(file) -> dict[str, list[District]]:
    data = pd.read_csv(file)
    map: dict[str, list[District]] = {
        normalize_town(jurisdiction): [] for jurisdiction in set(data["Jurisdiction"])
    }
    for i, row in data.iterrows():
        town = normalize_town(row["Jurisdiction"])
        district = District(
            full_name=row["Full District Name"],
            short_name=row["AbbreviatedDistrict"],
        )
        map[town].append(district)
    return map


def page_coverage(search_result: list[PageSearchOutput]) -> list[list[int]]:
    pages_covered = []
    for r in search_result:
        chunks = r.text.split("NEW PAGE ")
        pages = []
        for chunk in chunks[1:]:
            page = chunk.split("\n")[0]
            pages.append(int(page))
        pages_covered.append(pages)
    return pages_covered


def flatten(t: Iterable[Iterable[T]]) -> list[T]:
    return [item for sublist in t for item in sublist]


# cache = dc.Cache(get_project_root() / ".diskcache")


# @cached(cache, lambda *args: json.dumps(args))
@retry(
    retry=retry_if_exception_type(
        (
            APIError,
            RateLimitError,
            APIConnectionError,
            Timeout,
        )
    ),
    wait=wait_random_exponential(multiplier=1, max=60),
)
def semantic_comparison(true_answer: str, predicted: str) -> bool:
    client = OpenAI()
    template = Environment(
        loader=FileSystemLoader("zoning/llm/templates")
    ).get_template("semantic_comparison.pmpt.tpl")
    # TODO: Is there a way to share this implementation with our generic prompt
    # function?
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0.0,  # We want these responses to be deterministic
        max_tokens=1,
        messages=[
            {
                "role": "user",
                "content": template.render(
                    predicted=predicted,
                    true_answer=true_answer,
                ),
            },
        ],
    )
    top_choice = resp.choices[0]
    text = top_choice.message.content
    return text == "Y"
