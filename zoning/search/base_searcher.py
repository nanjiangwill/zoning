from abc import ABC, abstractmethod
from typing import Iterable

from zoning.class_types import SearchConfig, SearchQuery, SearchResult
from zoning.utils import get_thesaurus


class Searcher(ABC):
    def __init__(self, search_config: SearchConfig):
        self.search_config = search_config

    def expand_term(self, thesarus_file: str, eval_term: str) -> Iterable[str]:
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

    @abstractmethod
    def search(self, search_query: SearchQuery, target: str) -> SearchResult:
        pass
