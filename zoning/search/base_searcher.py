from abc import ABC, abstractmethod
from typing import Generator

from omegaconf import DictConfig

from ..utils import District, PageSearchOutput


class Searcher(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def search(
        self, town: str, district: District, term: str
    ) -> Generator[PageSearchOutput, None, None]:
        pass
