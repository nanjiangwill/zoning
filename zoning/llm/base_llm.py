import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import diskcache as dc
from dotenv import find_dotenv, load_dotenv
from jinja2 import Environment, FileSystemLoader
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError
from tenacity import retry, wait_random_exponential

from zoning.class_types import (
    LLMConfig,
    LLMInferenceResults,
    LLMQuery,
    Place,
    SearchResults,
)
from zoning.utils import cached, get_thesaurus, limit_global_concurrency

# dotenv will not override the env var if it's already set
load_dotenv(find_dotenv())


class LLM(ABC):
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.prompt_env = Environment(loader=FileSystemLoader(llm_config.templates_dir))
        self.cache_dir = dc.Cache(llm_config.cache_dir)
        extraction_chat_completion_tmpl = self.prompt_env.get_template(
            "extraction_chat_completion.pmpt.tpl"
        )
        extraction_completion_tmpl = self.prompt_env.get_template(
            "extraction_completion.pmpt.tpl"
        )

        # extract_chat_completion_tmpl = self.prompt_env.get_template(
        #     "extract_chat_completion.pmpt.tpl"
        # )

        self.TEMPLATE_MAPPING = {
            "text-davinci-003": extraction_completion_tmpl,
            "gpt-3.5-turbo": extraction_chat_completion_tmpl,
            "gpt-4": extraction_chat_completion_tmpl,
            "gpt-4-1106-preview": extraction_chat_completion_tmpl,
            "gpt-4-turbo": extraction_chat_completion_tmpl,
        }
        # Only support OPENAI for now
        self.aclient = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def get_prompt(
        self, place: Place, eval_term: str, searched_text: str
    ) -> List[Dict[str, str]] | str:
        synonyms = ", ".join(
            get_thesaurus(self.llm_config.thesaurus_file).get(eval_term, [])
        )
        match self.llm_config.llm_name:
            case "text-davinci-003":
                return self.TEMPLATE_MAPPING[self.llm_config.llm_name].render(
                    passage=searched_text,
                    term=eval_term,
                    synonyms=synonyms,
                    zone_name=place.district_full_name,
                    zone_abbreviation=place.district_short_name,
                )
            case "gpt-3.5-turbo" | "gpt-4":
                return [
                    {
                        "role": "system",
                        "content": self.TEMPLATE_MAPPING[
                            self.llm_config.llm_name
                        ].render(
                            term=eval_term,
                            synonyms=synonyms,
                            zone_name=place.district_full_name,
                            zone_abbreviation=place.district_short_name,
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Input: \n\n {searched_text}\n\n Output:",
                    },
                ]
            case "gpt-4-1106-preview" | "gpt-4-turbo":
                return [
                    {
                        "role": "system",
                        "content": self.TEMPLATE_MAPPING[
                            self.llm_config.llm_name
                        ].render(
                            term=eval_term,
                            synonyms=synonyms,
                            zone_name=place.district_full_name,
                            zone_abbreviation=place.district_short_name,
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Input: \n\n {searched_text}\n\n Output:",
                    },
                ]
            case _:
                raise ValueError(f"Unknown model name: {self.llm_config.llm_name}")

    # TODO: add back caching later
    # @(
    #     lambda method:
    #         lambda self, *args, **kwargs:
    #             cached(self.cache_dir, lambda *args, **kwargs: json.dumps(args) + json.dumps(kwargs))(method)(*args, **kwargs)
    # )
    @limit_global_concurrency(100)
    @retry(wait=wait_random_exponential(min=1, max=60))
    async def query_llm_once(
        self, llm_query: LLMQuery
    ) -> Tuple[List[Dict[str, str]], str | None]:
        input_prompt = self.get_prompt(
            llm_query.place, llm_query.eval_term, llm_query.search_match.text
        )
        base_params = {
            "model": self.llm_config.llm_name,
            "max_tokens": self.llm_config.max_tokens,
            "temperature": 0.0,
        }

        try:
            match self.llm_config.llm_name:
                case "text-davinci-003":
                    resp = await self.aclient.completions.create(
                        **base_params, prompt=input_prompt
                    )
                    top_choice = resp.choices[0]  # type: ignore
                    return input_prompt, top_choice.text
                case "gpt-3.5-turbo" | "gpt-4" | "gpt-4-turbo-preview" | "gpt-4-turbo":
                    resp = await self.aclient.chat.completions.create(
                        **base_params, messages=input_prompt
                    )
                    top_choice = resp.choices[0]  # type: ignore
                    return input_prompt, top_choice.message.content
                case "gpt-4-1106-preview":
                    if not self.llm_config.formatted_response:
                        resp = await self.aclient.chat.completions.create(
                            **base_params, messages=input_prompt
                        )
                        top_choice = resp.choices[0]  # type: ignore
                        return input_prompt, top_choice.message.content
                    else:
                        resp = await self.aclient.chat.completions.create(
                            **base_params,
                            messages=input_prompt,
                            response_format={"type": "json_object"},
                        )
                        top_choice = resp.choices[0]  # type: ignore
                        return input_prompt, top_choice.message.content
                case _:
                    raise ValueError(f"Unknown model name: {self.llm_config.llm_name}")
        except Exception as exc:
            print("Error running prompt", exc)
            # return input_prompt, None

    def parse_llm_output(self, model_response: str | None) -> dict | None:
        if model_response is None or model_response == "null":
            return None
        try:
            # TODO: this is something that came with new gpt update. This is a bandaid solution that i'll look into later
            if model_response[:7] == "```json":
                model_response = model_response[7:-4]
            json_body = json.loads(model_response)
            if json_body is None:
                # The model is allowed to return null if it cannot find the answer,
                # so just pass this onwards.
                return None
            return json_body
        except (ValidationError, TypeError, json.JSONDecodeError) as exc:
            print("Error parsing response from model during extraction:", exc)
            print(f"Response: {model_response}")
            return None

    @abstractmethod
    async def query(self, search_results: SearchResults) -> LLMInferenceResults:
        pass
