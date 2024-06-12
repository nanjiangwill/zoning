from asyncio import gather

from zoning.class_types import LLMInferenceResult, LLMQueries, LLMQuery
from zoning.llm.base_llm import LLM


class VanillaLLM(LLM):
    def __init__(self, config):
        super().__init__(config)

    async def query(self, llm_queries: LLMQueries) -> list[LLMInferenceResult]:
        async def worker(llm_query: LLMQuery) -> LLMInferenceResult:
            input_prompt, raw_model_response = await self.query_llm(llm_query)
            parsed_model_response = self.parse_llm_output(raw_model_response)

            if parsed_model_response is None:
                return LLMInferenceResult(
                    input_prompt=input_prompt, raw_model_response=raw_model_response
                )
            return LLMInferenceResult(
                input_prompt=input_prompt,
                raw_model_response=raw_model_response,
                **parsed_model_response
            )

        return await gather(*map(worker, llm_queries.query_list))
