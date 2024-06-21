from abc import ABC, abstractmethod

from zoning.class_types import OCRConfig


class Extractor(ABC):
    def __init__(self, ocr_config: OCRConfig):
        self.ocr_config = ocr_config

    @abstractmethod
    def extract(self, pdf_file: str, ocr_file: str) -> None:
        pass
