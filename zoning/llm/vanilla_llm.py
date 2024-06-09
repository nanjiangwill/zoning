from asyncio import gather

from class_types import LLMInferenceResult, LLMQueries, LLMQuery

from .base_llm import LLM


class VanillaLLM(LLM):
    def __init__(self, config):
        super().__init__(config)

    async def query(self, llm_queries: LLMQueries) -> list[LLMInferenceResult]:
        async def worker(llm_query: LLMQuery) -> LLMInferenceResult:
            input_prompt, model_response = await self.query_llm(llm_query)
            model_response = self.parse_llm_output(model_response)

            if model_response is None:
                return LLMInferenceResult(input_prompt=input_prompt)
            return LLMInferenceResult(input_prompt=input_prompt, **model_response)

        return await gather(*map(worker, llm_queries.query_list))
