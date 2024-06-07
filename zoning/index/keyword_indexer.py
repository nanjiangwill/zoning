from class_types import IndexEntities, IndexEntity
from elasticsearch import Elasticsearch
from omegaconf import DictConfig
from tqdm.contrib.concurrent import thread_map

from .base_indexer import Indexer


class KeywordIndexer(Indexer):
    def __init__(self, indexer_config: DictConfig):
        super().__init__(indexer_config)
        self.es_client = Elasticsearch(indexer_config.index.es_endpoint)

    def _index(self, index_entity: IndexEntity) -> None:
        for (
            index_data
        ) in (
            index_entity.get_index_data()
        ):  # get index data from one file, which contains multiple pages
            self.es_client.index(
                index=index_data.index,
                id=index_data.id,
                document=index_data.document,
                request_timeout=index_data.request_timeout,
            )

    def index(self, index_entities: IndexEntities) -> None:
        thread_map(self._index, index_entities.index_entities)
