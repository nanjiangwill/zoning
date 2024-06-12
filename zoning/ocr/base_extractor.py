from abc import ABC, abstractmethod

from omegaconf import DictConfig

from zoning.class_types import ExtractionEntities


class Extractor(ABC):
    def __init__(self, extractor_config: DictConfig):
        self.config = extractor_config
        if self.config.target_state == "all":
            raise NotImplementedError(
                "Post-extraction for all states not yet implemented."
            )

    @abstractmethod
    def extract(self, extract_targets: ExtractionEntities) -> None:
        pass
