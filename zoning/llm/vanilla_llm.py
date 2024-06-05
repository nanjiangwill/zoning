from asyncio import gather

from ..utils import LookupOutput, flatten, page_coverage
from .base_llm import LLM


class VanillaLLM(LLM):
    def __init__(self, config):
        super().__init__(config)

    async def query(self, town, district, term, pages):
        async def worker(page):
            prompt = self.get_prompt(town, district, term, page.text)
            return (
                page,
                self.parse_llm_output(await self.query_llm(prompt)),
            )

        for page, result in await gather(*map(worker, pages)):
            yield LookupOutput(
                output=result,
                search_pages=[page],
                search_pages_expanded=flatten(page_coverage([page])),
            )
