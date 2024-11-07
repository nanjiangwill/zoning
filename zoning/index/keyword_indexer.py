from elasticsearch import Elasticsearch

from zoning.class_types import ElasticSearchIndexData, FormatOCR, IndexConfig
from zoning.index.base_indexer import Indexer


class KeywordIndexer(Indexer):
    def __init__(self, index_config: IndexConfig):
        super().__init__(index_config)
        self.es_client = Elasticsearch(index_config.es_endpoint)

    def index(self, formatted_ocr: FormatOCR, town: str) -> None:
        all_index_data = []
        page_data = formatted_ocr.pages

        for idx in range(len(page_data)):
            text = ""
            for j in range(self.index_config.index_range):
                if idx + j >= len(page_data):
                    break
                text += (
                    f"\nNEW PAGE {page_data[idx + j]['page']}\n"
                    + page_data[idx + j]["text"]
                )
            all_index_data.append(
                ElasticSearchIndexData(
                    index=town,
                    id=page_data[idx]["page"],
                    document={"Page": str(page_data[idx]["page"]), "Text": text},
                    request_timeout=30,
                )
            )
        # get index data from one file, which contains multiple pages
        for index_data in all_index_data:
            self.es_client.index(
                index=index_data.index,
                id=index_data.id,
                document=index_data.document,
                request_timeout=index_data.request_timeout,
            )
