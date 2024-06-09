from asyncio import gather

from class_types import LLMInferenceResult, LLMQueries, LLMQuery

from .base_llm import LLM


class VanillaLLM(LLM):
    def __init__(self, config):
        super().__init__(config)

    async def query(self, llm_queries: LLMQueries) -> list[LLMInferenceResult]:
        async def worker(llm_query: LLMQuery):
            return self.parse_llm_output(await self.query_llm(llm_query))

        return await gather(*map(worker, llm_queries.query_list))
