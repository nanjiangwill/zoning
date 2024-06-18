from abc import ABC, abstractmethod

from zoning.class_types import IndexConfig, IndexEntities, OCREntities


class Indexer(ABC):
    def __init__(self, index_config: IndexConfig):
        self.index_config = index_config

    @abstractmethod
    def index(self, ocr_entities: OCREntities) -> IndexEntities:
        pass
