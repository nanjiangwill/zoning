import asyncio
from asyncio import gather

import tqdm

from zoning.class_types import (
    LLMInferenceResult,
    LLMInferenceResults,
    LLMOutput,
    LLMQuery,
    SearchResult,
    SearchResults,
)
from zoning.llm.base_llm import LLM
from zoning.utils import flatten, page_coverage


class VanillaLLM(LLM):
    def __init__(self, config):
        super().__init__(config)

    async def query_llm(self, llm_query: LLMQuery) -> LLMOutput:
        input_prompt, raw_model_response = await self.query_llm_once(llm_query)
        parsed_model_response = self.parse_llm_output(raw_model_response)

        return LLMOutput(
            place=llm_query.place,
            eval_term=llm_query.eval_term,
            search_match=llm_query.search_match,
            input_prompt=input_prompt,
            search_page_range=sorted(
                flatten(page_coverage([llm_query.search_match.text]))
            ),
            raw_model_response=raw_model_response,
            **parsed_model_response if parsed_model_response else {},
        )

    async def query_one_place_one_eval(
        self, search_result: SearchResult
    ) -> LLMInferenceResult:
        llm_queries = []

        for search_match in search_result.search_matches:
            llm_queries.append(
                LLMQuery(
                    place=search_result.place,
                    eval_term=search_result.eval_term,
                    search_match=search_match,
                )
            )
        llm_outputs = await gather(*map(self.query_llm, llm_queries))
        return LLMInferenceResult(
            place=search_result.place,
            eval_term=search_result.eval_term,
            search_result=search_result,
            llm_inference_result=llm_outputs,
        )

    async def query(self, search_results: SearchResults) -> LLMInferenceResults:
        async_tasks = [
            self.query_one_place_one_eval(search_result)
            for search_result in search_results.search_results
        ]
        async_results = []

        pbar = tqdm.tqdm(total=len(async_tasks))
        for f in asyncio.as_completed(async_tasks):
            async_result = await f
            pbar.update()
            if async_result is not None:
                pbar.set_description(
                    f"Processing {async_result.place} {async_result.eval_term}"
                )
                async_results.append(async_result)
        pbar.close()

        return LLMInferenceResults(llm_inference_results=async_results)
