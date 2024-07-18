from abc import ABC
from typing import List

from jinja2 import Environment, FileSystemLoader

from zoning.class_types import Prompt, PromptResult, SearchMatch, SearchResult
from zoning.utils import get_thesaurus, page_coverage_text, prompt_file


class PromptGenerator(ABC):
    def __init__(self, prompt_config):
        self.prompt_config = prompt_config
        self.prompt_env = Environment(
            loader=FileSystemLoader(prompt_config.templates_dir)
        )
        self.prompt_template = self.prompt_env.get_template(
            prompt_file(prompt_config.method)
        )

    def merge_overlapping_searches(
        self, search_lists: List[SearchMatch]
    ) -> List[List[SearchMatch]]:
        # print(search_lists.)
        if len(search_lists) == 0:
            return []

        sorted_lists = sorted(search_lists, key=lambda x: x.page_range[0])

        merged = [[sorted_lists[0]]]

        for current in sorted_lists[1:]:
            last = merged[-1]
            if current.page_range[0] <= last[-1].page_range[-1]:
                merged[-1].append(current)
            else:
                merged.append([current])

        return merged

    def generate_prompt(self, search_result: SearchResult, target: str) -> PromptResult:
        synonyms = ", ".join(
            get_thesaurus(self.prompt_config.thesaurus_file).get(
                search_result.eval_term, []
            )
        )

        system_prompt = self.prompt_template.render(
            term=search_result.eval_term,
            synonyms=synonyms,
            zone_name=search_result.place.district_full_name,
            zone_abbreviation=search_result.place.district_short_name,
        )

        # we construct the user prompt for each search match
        if self.prompt_config.merge_search_matches:
            merged_searches = self.merge_overlapping_searches(
                search_result.search_matches
            )
            merged_texts = [
                page_coverage_text([i.text for i in ms]) for ms in merged_searches
            ]
            all_prompts = [
                Prompt(
                    system_prompt=system_prompt,
                    user_prompt=f"Input: \n\n {merged_text}\n\n Output:",
                )
                for merged_text in merged_texts
            ]
        else:
            user_prompts = [
                f"Input: \n\n {i.text}\n\n Output:"
                for i in search_result.search_matches
            ]

            all_prompts = [
                Prompt(system_prompt=system_prompt, user_prompt=user_prompts[i])
                for i in range(len(search_result.search_matches))
            ]

        return PromptResult(
            place=search_result.place,
            eval_term=search_result.eval_term,
            search_result=search_result,
            input_prompts=all_prompts,
            merge_text=self.prompt_config.merge_search_matches,
            merge_text_matches=(
                merged_searches if self.prompt_config.merge_search_matches else None
            ),
        )
