from asyncio import gather
from typing import AsyncGenerator

from class_types import LLMInferenceResult, LLMQueries, LLMQuery

from .base_llm import LLM


class VanillaLLM(LLM):
    def __init__(self, config):
        super().__init__(config)

    async def query(
        self, llm_queries: LLMQueries
    ) -> AsyncGenerator[LLMInferenceResult, None]:
        async def worker(llm_query: LLMQuery):
            return self.parse_llm_output(await self.query_llm(llm_query))

        for llm_inference_result in await gather(*map(worker, llm_queries)):
            yield llm_inference_result
