from abc import ABC, abstractmethod

from zoning.class_types import OCRConfig, OCREntities


class Extractor(ABC):
    def __init__(self, ocr_config: OCRConfig):
        self.ocr_config = ocr_config

    @abstractmethod
    def extract(self, ocr_entities: OCREntities) -> None:
        pass
