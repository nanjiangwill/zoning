import numpy as np
from datasets import Dataset
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from tenacity import retry, wait_random_exponential

from zoning.class_types import (
    DistrictExtractionConfig,
    DistrictExtractionResult,
    DistrictExtractionVerificationResult,
    FormatOCR,
    PageEmbeddingResult,
)
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

        self.verification_es_client = Elasticsearch(
            self.district_extraction_config.verification_es_endpoint
        )

    @retry(wait=wait_random_exponential(min=1, max=60))
    def call_llm(self, system_prompt: str, user_prompt: str):
        call_response = self.openai_client.chat.completions.create(
            model=self.district_extraction_config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return call_response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60))
    def page_embedding(self, x: FormatOCR, target: str) -> PageEmbeddingResult:
        town = x.town
        pages = x.pages

        y = []
        for p in pages:
            p_text = p["text"]
            if not p_text:
                p_text = " "
            if len(p_text.split()) > 3000:
                p_text = " ".join(p_text[:3000])
            y.append(p_text)

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
            town=town,
            embedded_pages=embedded_pages,
        )

    def district_extraction(
        self, x: PageEmbeddingResult, target: str
    ) -> DistrictExtractionResult:
        town = x.town
        embedded_pages = x.embedded_pages

        dataset = Dataset.from_list(embedded_pages)
        dataset.add_faiss_index("embedding")
        k = 2

        if self.QUERY_EMBEDDING is None:
            query_embedding_response = self.openai_client.embeddings.create(
                input=self.QUERY, model=self.district_extraction_config.embedding_model
            )

            self.QUERY_EMBEDDING = np.array(query_embedding_response.data[0].embedding)

        res = dataset.get_nearest_examples("embedding", self.QUERY_EMBEDDING, k)

        nearest_pages = res.examples["page"]

        # extend nearest_page by +- extend_range
        extend_range = 1  # 2
        extended_nearest_pages = flatten(
            [
                [int(i) + j for j in range(-extend_range, extend_range + 1)]
                for i in nearest_pages
            ]
        )
        extended_nearest_pages = sorted(list(set(extended_nearest_pages)))

        districts = []

        for i in range(len(extended_nearest_pages) - 1):
            if int(extended_nearest_pages[i + 1]) - int(extended_nearest_pages[i]) > 1:
                continue
            query_page = [
                int(extended_nearest_pages[i]),
                int(extended_nearest_pages[i + 1]),
            ]

            query_texts = [
                i["text"] for i in embedded_pages if int(i["page"]) in query_page
            ]

            district_extraction_response = self.call_llm(
                self.system_prompt_template.render(),
                self.user_prompt_template.render(docs=query_texts),
            )

            districts.extend(post_processing_llm_output(district_extraction_response))

        unique_districts = set(frozenset(d.items()) for d in districts)
        unique_districts = [dict(d) for d in unique_districts]
        # extended_nearest_texts = [
        #     i["text"] for i in embedded_pages if int(i["page"]) in extended_nearest_pages
        # ]

        # district_extraction_response = self.call_llm(
        #     self.system_prompt_template.render(),
        #     self.user_prompt_template.render(docs=extended_nearest_texts),
        # )

        # return DistrictExtractionResult(
        #     town=town,
        #     districts=post_processing_llm_output(district_extraction_response),
        #     districts_info_page=extended_nearest_pages,
        # )
        return DistrictExtractionResult(
            town=town,
            districts=unique_districts,
            districts_info_page=extended_nearest_pages,
        )

    def get_district_query(
        self,
        district_full_name: str,
        district_short_name: str,
        boost_value: float = 1.0,
    ):
        return (
            Q("match_phrase", Text={"query": district_full_name, "boost": boost_value})
            | Q(
                "match_phrase",
                Text={"query": district_short_name, "boost": boost_value},
            )
            | Q(
                "match_phrase",
                Text={
                    "query": district_short_name.replace("-", ""),
                    "boost": boost_value,
                },
            )
            | Q(
                "match_phrase",
                Text={
                    "query": district_short_name.replace(".", ""),
                    "boost": boost_value,
                },
            )
        )

    def district_extraction_verification(
        self, x: DistrictExtractionResult, target: str
    ) -> DistrictExtractionVerificationResult:

        valid_districts = []

        for d in x.districts:
            district_full_name = d["T"]
            district_short_name = d["Z"]
            district_query = self.get_district_query(
                district_full_name, district_short_name
            )

            s = Search(using=self.verification_es_client, index=x.town)
            s.query = district_query
            s = s.extra(size=1)
            s = s.highlight("Text")
            res = s.execute()

            if len(res) == 0:
                print(f"No results found for {d['T']}-{d['Z']}")
            else:
                valid_districts.append(f"{x.town}__{d['Z']}__{d['T']}")

        return DistrictExtractionVerificationResult(
            town=x.town,
            valid_districts=valid_districts,
            districts_info_page=x.districts_info_page,
        )