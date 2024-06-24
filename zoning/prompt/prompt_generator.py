from abc import ABC

from jinja2 import Environment, FileSystemLoader

from zoning.class_types import Prompt, PromptResult, SearchResult
from zoning.utils import get_thesaurus, prompt_file


class PromptGenerator(ABC):
    def __init__(self, prompt_config):
        self.prompt_config = prompt_config
        self.prompt_env = Environment(
            loader=FileSystemLoader(prompt_config.templates_dir)
        )
        self.prompt_template = self.prompt_env.get_template(
            prompt_file(prompt_config.method)
        )

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
        user_prompts = [
            f"Input: \n\n {i.text}\n\n Output:" for i in search_result.search_matches
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
        )
