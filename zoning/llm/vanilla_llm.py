from asyncio import gather
from typing import AsyncGenerator

from class_types import LLMInferenceResult, LLMQuery, SearchResult

from .base_llm import LLM


class VanillaLLM(LLM):
    def __init__(self, config):
        super().__init__(config)

    async def query(
        self, llm_query: LLMQuery
    ) -> AsyncGenerator[LLMInferenceResult, None]:
        async def worker(search_result: SearchResult):
            return (self.parse_llm_output(await self.query_llm(search_result)),)

        for llm_inference_result in await gather(
            *map(worker, llm_query.search_results)
        ):
            yield llm_inference_result
