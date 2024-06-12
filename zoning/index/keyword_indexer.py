import json

from elasticsearch import Elasticsearch
from omegaconf import DictConfig
from tqdm.contrib.concurrent import thread_map

from zoning.class_types import ElasticSearchIndexData, IndexEntities, IndexEntity
from zoning.index.base_indexer import Indexer


class KeywordIndexer(Indexer):
    def __init__(self, indexer_config: DictConfig):
        super().__init__(indexer_config)
        self.es_client = Elasticsearch(indexer_config.index.es_endpoint)

    def _index(self, index_entity: IndexEntity) -> None:
        all_index_data = []
        with open(index_entity.dataset_file, "r") as f:
            data = json.load(f)
        for idx in range(len(data)):
            text = ""
            for j in range(index_entity.index_range):
                if idx + j >= len(data):
                    break
                text += f"\nNEW PAGE {idx + j + 1}\n" + data[idx + j]["Text"]
            all_index_data.append(
                ElasticSearchIndexData(
                    index=index_entity.name,
                    id=str(idx + 1),
                    document={"Page": str(idx + 1), "Text": text},
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

    def index(self, index_entities: IndexEntities) -> None:
        thread_map(self._index, index_entities.index_entities)
