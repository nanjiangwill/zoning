from datasets import Dataset
from elasticsearch import Elasticsearch
from omegaconf import DictConfig
from tqdm.contrib.concurrent import thread_map

from .base_indexer import Indexer


class KeywordIndexer(Indexer):
    def __init__(self, indexer_config: DictConfig):
        super().__init__(indexer_config)

    def _index_town(self, es_client: Elasticsearch, town: str, dataset: Dataset):
        town_dataset = dataset.filter(lambda example: example["Town"] == town)

        for page_idx in range(len(town_dataset)):
            text = ""

            for j in range(self.config.index.index_page_range):
                if page_idx + j >= len(town_dataset):
                    break
                text += (
                    f"\nNEW PAGE {page_idx + j + 1}\n"
                    + town_dataset[page_idx + j]["Text"]
                )

            # Truncate to 2000 tokens
            # text = enc.decode(enc.encode(text)[:2500])

            es_client.index(
                index=town,
                id=page_idx + 1,
                document={"Page": page_idx + 1, "Text": text},
                request_timeout=30,
            )

    def index(self, es_client: Elasticsearch, dataset: Dataset):
        all_towns = set(dataset["Town"])
        thread_map(lambda town: self._index_town(es_client, town, dataset), all_towns)
