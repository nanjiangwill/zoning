import os
from abc import ABC
from typing import Dict, List, Tuple

import diskcache as dc
from anthropic import AsyncAnthropic
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, wait_random_exponential

from zoning.class_types import LLMConfig, LLMInferenceResult, LLMOutput, PromptResult
from zoning.utils import cached, limit_global_concurrency, post_processing_llm_output

# dotenv will not override the env var if it's already set
load_dotenv(find_dotenv())


class LLM(ABC):
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.cache_dir = dc.Cache(llm_config.cache_dir)

        if self.llm_config.llm_name in [
            "gpt-4-1106-preview",
            "gpt-4-turbo",
            "text-davinci-003",
        ]:
            self.aclient = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif self.llm_config.llm_name in [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]:
            self.aclient = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def get_prompt(
        self, system_prompt: str, user_prompt: str
    ) -> List[Dict[str, str]] | str:
        match self.llm_config.llm_name:
            case "text-davinci-003":
                return f"{system_prompt}\n\n{user_prompt}"
            case "gpt-3.5-turbo" | "gpt-4" | "gpt-4-1106-preview" | "gpt-4-turbo":
                return [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ]
            case (
                "claude-3-5-sonnet-20240620"
                | "claude-3-opus-20240229"
                | "claude-3-sonnet-20240229"
                | "claude-3-haiku-20240307"
            ):
                return {
                    "system_prompt": system_prompt,
                    "user_prompt": [
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ],
                }
            case _:
                raise ValueError(f"Unknown model name: {self.llm_config.llm_name}")

    # TODO: add back caching later
    # @(
    #     lambda method:
    #         lambda self, *args, **kwargs:
    #             cached(self.cache_dir, lambda *args, **kwargs: json.dumps(args) + json.dumps(kwargs))(method)(*args, **kwargs)
    # )
    # @limit_global_concurrency(100)
    @retry(wait=wait_random_exponential(min=1, max=60))
    async def call_llm(
        self, input_prompt: List[Dict[str, str]] | str
    ) -> Tuple[List[Dict[str, str]], str | None]:
        base_params = {
            "model": self.llm_config.llm_name,
            "max_tokens": self.llm_config.max_tokens,
            "temperature": 0.0,
        }

        match self.llm_config.llm_name:
            case "text-davinci-003":
                resp = await self.aclient.completions.create(
                    **base_params, prompt=input_prompt
                )
                top_choice = resp.choices[0]  # type: ignore
                model_response = top_choice.text

            case "gpt-3.5-turbo" | "gpt-4" | "gpt-4-turbo-preview" | "gpt-4-turbo":
                resp = await self.aclient.chat.completions.create(
                    **base_params, messages=input_prompt
                )
                top_choice = resp.choices[0]  # type: ignore
                model_response = top_choice.message.content
            case "gpt-4-1106-preview":
                if not self.llm_config.formatted_response:
                    resp = await self.aclient.chat.completions.create(
                        **base_params, messages=input_prompt
                    )
                    top_choice = resp.choices[0]  # type: ignore
                    model_response = top_choice.message.content
                else:
                    resp = await self.aclient.chat.completions.create(
                        **base_params,
                        messages=input_prompt,
                        response_format={"type": "json_object"},
                    )
                    top_choice = resp.choices[0]  # type: ignore
                    model_response = top_choice.message.content
            case (
                "claude-3-5-sonnet-20240620"
                | "claude-3-opus-20240229"
                | "claude-3-sonnet-20240229"
                | "claude-3-haiku-20240307"
            ):
                resp = await self.aclient.messages.create(
                    **base_params,
                    system=input_prompt["system_prompt"],
                    messages=input_prompt["user_prompt"],
                )
                top_choice = resp.content[0]  # type: ignore
                model_response = top_choice.text
            case _:
                raise ValueError(f"Unknown model name: {self.llm_config.llm_name}")

        if model_response == "null" or model_response is None:
            print("Model response is:", model_response)
            print("Retrying")
            raise ValueError(f"Model response is null: {model_response}")
        else:
            return input_prompt, model_response

    async def query(
        self, prompt_result: PromptResult, target: str
    ) -> LLMInferenceResult:
        llm_outputs = []
        # we query the llm for each search match
        for ip_idx in range(len(prompt_result.input_prompts)):
            system_prompt = prompt_result.input_prompts[ip_idx].system_prompt
            user_prompt = prompt_result.input_prompts[ip_idx].user_prompt
            if user_prompt == "Input: \n\n \n\n Output:":
                continue
            # we query the llm
            input_prompt, raw_model_response = await self.call_llm(
                self.get_prompt(system_prompt, user_prompt)
            )
            parsed_model_response = post_processing_llm_output(raw_model_response)
            llm_output = LLMOutput(
                place=prompt_result.place,
                eval_term=prompt_result.eval_term,
                raw_model_response=raw_model_response,
                **parsed_model_response if parsed_model_response else {},
            )
            llm_outputs.append(llm_output)

        return LLMInferenceResult(
            place=prompt_result.place,
            eval_term=prompt_result.eval_term,
            llm_outputs=llm_outputs,
        )
