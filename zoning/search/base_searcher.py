from omegaconf import DictConfig
from ..utils import District


class Searcher:
    def __init__(self, config: DictConfig):
        self.config = config

    def search(self, town: str, district: District, term: str):
        pass
