import json
from abc import ABC, abstractmethod
from typing import AsyncGenerator

from jinja2 import Environment, FileSystemLoader
from omegaconf import DictConfig
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from ..utils import District, ExtractionOutput, LookupOutput, get_thesaurus


class LLM(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
        self.prompt_env = Environment(loader=FileSystemLoader("zoning/llm/templates"))
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
        self.aclient = AsyncOpenAI()
        self.client = OpenAI()

    def get_prompt(
        self, town: str, district: District, term: str, page_text: str
    ) -> list[dict[str, str]] | str:
        synonyms = ", ".join(get_thesaurus(self.config.thesaurus_file).get(term, []))
        match self.config.llm.model_name:
            case "text-davinci-003":
                return self.TEMPLATE_MAPPING[self.config.llm.model_name].render(
                    passage=page_text,
                    term=term,
                    synonyms=synonyms,
                    zone_name=district.full_name,
                    zone_abbreviation=district.short_name,
                )
            case "gpt-3.5-turbo" | "gpt-4":
                return [
                    {
                        "role": "system",
                        "content": self.TEMPLATE_MAPPING[
                            self.config.llm.model_name
                        ].render(
                            term=term,
                            synonyms=synonyms,
                            zone_name=district.full_name,
                            zone_abbreviation=district.short_name,
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Input: \n\n {page_text}\n\n Output:",
                    },
                ]
            case "gpt-4-1106-preview" | "gpt-4-turbo":
                return [
                    {
                        "role": "system",
                        "content": self.TEMPLATE_MAPPING[
                            self.config.llm.model_name
                        ].render(
                            term=term,
                            synonyms=synonyms,
                            zone_name=district.full_name,
                            zone_abbreviation=district.short_name,
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Input: \n\n {page_text}\n\n Output:",
                    },
                ]
            case _:
                raise ValueError(f"Unknown model name: {self.config.llm.model_name}")

    # @cached(cache, lambda *args, **kwargs: json.dumps(args) + json.dumps(kwargs))
    # @limit_global_concurrency(100)
    async def query_llm(
        self,
        input_prompt: str | list[dict[str, str]],
    ) -> str | None:
        # raise NotImplementedError
        base_params = {
            "model": self.config.llm.model_name,
            "max_tokens": self.config.llm.max_tokens,
            "temperature": 0.0,
        }

        try:
            match self.config.llm.model_name:
                case "text-davinci-003":
                    resp = await self.aclient.completions.create(
                        **base_params, prompt=input_prompt
                    )
                    top_choice = resp.choices[0]  # type: ignore
                    return top_choice.text
                case "gpt-3.5-turbo" | "gpt-4" | "gpt-4-turbo-preview" | "gpt-4-turbo":
                    resp = await self.aclient.chat.completions.create(
                        **base_params, messages=input_prompt
                    )
                    top_choice = resp.choices[0]  # type: ignore
                    return top_choice.message.content
                case "gpt-4-1106-preview":
                    if not self.config.llm.formatted_response:
                        resp = await self.aclient.chat.completions.create(
                            **base_params, messages=input_prompt
                        )
                        top_choice = resp.choices[0]  # type: ignore
                        return top_choice.message.content
                    else:
                        resp = await self.aclient.chat.completions.create(
                            **base_params,
                            messages=input_prompt,
                            response_format={"type": "json_object"},
                        )
                        top_choice = resp.choices[0]  # type: ignore
                        return top_choice.message.content
                case _:
                    raise ValueError(
                        f"Unknown model name: {self.config.llm.model_name}"
                    )
        except Exception as exc:
            print("Error running prompt", exc)
            return None

    def parse_llm_output(self, output: str | None) -> ExtractionOutput | None:
        if output is None or output == "null":
            return None

        try:
            # TODO: this is something that came with new gpt update. This is a bandaid solution that i'll look into later
            if output[:7] == "```json":
                output = output[7:-4]
            json_body = json.loads(output)
            if json_body is None:
                # The model is allowed to return null if it cannot find the answer,
                # so just pass this onwards.
                return None
            return ExtractionOutput(**json_body)
        except (ValidationError, TypeError, json.JSONDecodeError) as exc:
            print("Error parsing response from model during extraction:", exc)
            print(f"Response: {output}")
            return None

    @abstractmethod
    async def query(
        self, town, district, term, pages
    ) -> AsyncGenerator[LookupOutput, None]:
        pass
