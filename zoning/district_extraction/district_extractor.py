import json

import numpy as np
from datasets import Dataset
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

from zoning.class_types import (
    DistrictExtractionConfig,
    DistrictExtractionResult,
    FormatOCR,
    PageEmbeddingResult,
)
from zoning.search.keyword_searcher import KeywordSearcher
from zoning.utils import flatten, post_processing_llm_output, prompt_file


class DistrictExtractor:
    def __init__(self, district_extraction_config: DistrictExtractionConfig):
        self.district_extraction_config = district_extraction_config

        self.QUERY = "Districts. The town is divided into the following district zones: * residential (R2-0) * industrial (I-10) * rural overlay (T-190)"
        self.QUERY_EMBEDDING = None

        self.openai_client = OpenAI()

        self.prompt_env = Environment(
            loader=FileSystemLoader(district_extraction_config.templates_dir)
        )
        self.system_prompt_template = self.prompt_env.get_template(
            prompt_file(district_extraction_config.system_prompt_file)
        )
        self.user_prompt_template = self.prompt_env.get_template(
            prompt_file(district_extraction_config.user_prompt_file)
        )

    def call_llm(self, system_prompt: str, user_prompt: str):
        call_response = self.openai_client.chat.completions.create(
            model=self.district_extraction_config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return call_response.choices[0].message.content

    # @retry(wait=wait_random_exponential(min=1, max=60))
    def page_embedding(self, x: FormatOCR, target: str) -> PageEmbeddingResult:
        town_name = x.town_name
        pages = x.pages

        y = []
        for p in pages:
            p_text = p["text"]
            if not p_text:
                p_text = " "
            if len(p_text.split()) > 3000:
                p_text = " ".join(p_text[:3000])
            y.append(p_text)

        print(f"Processing {town_name}")

        emb = self.openai_client.embeddings.create(
            input=y, model=self.district_extraction_config.embedding_model
        )
        assert len(emb.data) == len(pages)

        embedded_pages = [
            {
                "page": pages[i]["page"],
                "text": pages[i]["text"],
                "embedding": emb.data[i].embedding,
            }
            for i in range(len(pages))
        ]

        return PageEmbeddingResult(
            town_name=x.town_name,
            embedded_pages=embedded_pages,
        )

    def district_extraction(
        self, x: PageEmbeddingResult, target: str
    ) -> DistrictExtractionResult:
        town_name = x.town_name
        embedded_pages = x.embedded_pages

        dataset = Dataset.from_list(embedded_pages)
        dataset.add_faiss_index("embedding")
        k = 4

        if self.QUERY_EMBEDDING is None:
            query_embedding_response = self.openai_client.embeddings.create(
                input=self.QUERY, model=self.district_extraction_config.embedding_model
            )

            self.QUERY_EMBEDDING = np.array(query_embedding_response.data[0].embedding)

        res = dataset.get_nearest_examples("embedding", self.QUERY_EMBEDDING, k)

        nearest_pages = res.examples["page"]
        nearest_texts = res.examples["text"]

        districts_per_page = []
        for page, text in zip(nearest_pages, nearest_texts):
            district_extraction_response = self.call_llm(
                self.system_prompt_template.render(),
                self.user_prompt_template.render(docs=[text]),
            )
            districts_per_page.append(
                {
                    "page": page,
                    "districts": post_processing_llm_output(
                        district_extraction_response
                    ),
                }
            )

        districts = flatten([d["districts"] for d in districts_per_page])

        unique_districts = set(frozenset(d.items()) for d in districts)
        unique_districts = [dict(d) for d in unique_districts]

        return DistrictExtractionResult(
            town_name=town_name,
            districts=unique_districts,
            districts_info_page=nearest_pages,
        )

    def district_extraction_verification(
        self, x: DistrictExtractionResult, target: str
    ) -> None:

        # how to filter  same Z todo
        # use search engine to filter?
        valid_districts = []
        for d in x.districts:
            district_full_name = d["T"]
            district_short_name = d["Z"]
            district_query = KeywordSearcher.get_district_query(
                district_full_name, district_short_name, False
            )

            s = Search(using=self.es_client, index=search_query.place.town)
            s.query = district_query
            s = s.extra(size=self.search_config.num_results)
            s = s.highlight("Text")
            res = s.execute()

            if len(res) == 0:
                print(f"No results found for {d['T']}-{d['Z']}")
            else:
                valid_districts.append(f"{search_query.place.town}__{d['Z']}__{d['T']}")
        with open(f"{target}/valid_districts.json", "w") as f:
            json.dump(valid_districts, f, indent=4)
