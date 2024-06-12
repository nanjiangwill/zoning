import asyncio
import inspect
import json
import os
import re
from functools import partial, wraps
from typing import Iterable, TypeVar

from datasets import load_dataset
from elasticsearch_dsl import Q
from jinja2 import Environment, FileSystemLoader
from openai import APIConnectionError, APIError, OpenAI, RateLimitError, Timeout
from tenacity import retry, retry_if_exception_type, wait_random_exponential
from typer import Typer

T = TypeVar("T")


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


def expand_term(thesarus_file: str, eval_term: str) -> Iterable[str]:
    # term = term.replace("_", " ").strip()
    # logger.info(f"Term: {term}")  # Initial logging of the term
    thesarus = get_thesaurus(thesarus_file)
    min_variations = thesarus.get("min", [])
    max_variations = thesarus.get("max", [])
    expanded_count = 0
    for query in thesarus.get(
        eval_term, []
    ):  # Iterate over thesaurus entries for the term
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


def page_coverage(searched_text: list[str]) -> list[list[int]]:
    pages_covered = []
    for text in searched_text:
        chunks = text.split("NEW PAGE ")
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


def publish_dataset(extraction_target, config):
    hf_dataset_files = [
        os.path.join(extraction_target.dataset_dir, file)
        for file in os.listdir(extraction_target.dataset_dir)
    ]

    page_dataset = load_dataset("json", data_files=hf_dataset_files)

    if config.extract.hf_dataset.publish_dataset:
        page_dataset.push_to_hub(
            config.extract.hf_dataset.name,
            private=config.extract.hf_dataset.private,
        )


def get_district_query(
    district_full_name: str,
    district_short_name: str,
    is_district_fuzzy: bool,
    boost_value: float = 1.0,
) -> Q:
    exact_district_query = (
        Q(
            "match_phrase",
            Text={"query": district_full_name, "boost": boost_value},
        )
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
        district_query = Q("bool", should=[exact_district_query, fuzzy_district_query])
    else:
        district_query = exact_district_query

    return district_query


def get_eval_term_query(
    eval_term: str, is_eval_term_fuzzy: bool, thesaurus_file: str
) -> Q:
    expanded_eval_term = expand_term(thesaurus_file, eval_term)
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


def get_units_query(eval_term: str, thesaurus_file: str) -> Q:
    expanded_units = expand_term(thesaurus_file, f"{eval_term} units")
    units_query = Q(
        "bool",
        should=list(Q("match_phrase", Text=t) for t in expanded_units),
        minimum_should_match=1,
    )
    return units_query
