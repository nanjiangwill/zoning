from abc import ABC, abstractmethod
from omegaconf import DictConfig
from class_types import ExtractionTargetCollection

class Extractor(ABC):
    def __init__(self, extractor_config: DictConfig):
        self.config = extractor_config
        if self.config.target_state == "all":
            raise NotImplementedError(
                "Post-extraction for all states not yet implemented."
            )

    @abstractmethod
    def extract(self, extract_target: ExtractionTargetCollection) -> None:
        pass