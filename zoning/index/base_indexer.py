from abc import ABC, abstractmethod

from zoning.class_types import FormatOCR, IndexConfig


class Indexer(ABC):
    def __init__(self, index_config: IndexConfig):
        self.index_config = index_config

    @abstractmethod
    def index(self, format_ocr: FormatOCR, target: str) -> None:
        pass
