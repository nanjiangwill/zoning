from zoning.class_types import IndexConfig, IndexEntities, OCREntities
from zoning.index.base_indexer import Indexer


class EmbeddingIndexer(Indexer):
    def __init__(self, index_config: IndexConfig):
        super().__init__(index_config)

    def index(self, ocr_entities: OCREntities) -> IndexEntities:
        pass
