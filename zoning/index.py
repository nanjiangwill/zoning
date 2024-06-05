import os
from typing import cast

import hydra
from datasets import DatasetDict, load_from_disk
from elasticsearch import Elasticsearch
from indexer import KeywordIndexer
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    match config.index.method:
        case "keyword":
            indexer = KeywordIndexer(config)
        case "embdeding":
            raise NotImplementedError  # indexer = EmbeddingIndexer(config)
        case _:
            raise ValueError(f"Extractor {config.extract.name} not implemented")

    # TODO, merge output_dir and target_state to global variable
    dataset_path = os.path.join(
        config.data_output_dir, config.target_state, "hf_dataset"
    )
    dataset = load_from_disk(dataset_path)
    dataset = cast(DatasetDict, dataset)

    es_client = Elasticsearch(config.index.es_endpoint)

    # TODO, currently we do not split the dataset, we index the whole dataset, but load_dataset need to specify train/test, so we store everything in train
    indexer.index(es_client, dataset=dataset["train"])
    # indexer.index(dataset=dataset['train'])


if __name__ == "__main__":
    main()
