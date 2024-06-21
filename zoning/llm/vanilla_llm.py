from zoning.class_types import LLMInferenceResult, LLMOutput, LLMQuery, SearchResult
from zoning.llm.base_llm import LLM


class VanillaLLM(LLM):
    def __init__(self, config):
        super().__init__(config)

    async def query(
        self, search_result: SearchResult, target: str
    ) -> LLMInferenceResult:
        llm_outputs = []

        # we query the llm for each search match
        for search_match in search_result.search_matches:
            llm_query = LLMQuery(
                place=search_result.place,
                eval_term=search_result.eval_term,
                context=search_match.text,
            )

            # we query the llm
            input_prompt, raw_model_response = await self.call_llm(llm_query)
            parsed_model_response = self.parse_llm_output(raw_model_response)
            llm_output = LLMOutput(
                place=llm_query.place,
                eval_term=llm_query.eval_term,
                search_match=search_match,
                input_prompt=input_prompt,
                raw_model_response=raw_model_response,
                **parsed_model_response if parsed_model_response else {},
            )
            llm_outputs.append(llm_output)

        return LLMInferenceResult(
            place=search_result.place,
            eval_term=search_result.eval_term,
            search_result=search_result,
            llm_outputs=llm_outputs,
        )
