import os
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_from_disk, DatasetDict
from indexer import *
from typing import cast
from elasticsearch import Elasticsearch


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    match config.index.method:
        case "keyword":
            indexer = KeywordIndexer(config)
        case "embdeding":
            indexer = EmbeddingIndexer(config)
        case _:
            raise ValueError(f"Extractor {config.extract.name} not implemented")

    # TODO, merge output_dir and target_state to global variable
    dataset_path = os.path.join(
        config.index.output_dir, config.index.target_state, "hf_dataset"
    )
    dataset = load_from_disk(dataset_path)
    dataset = cast(DatasetDict, dataset)

    es = Elasticsearch(config.index.es_endpoint)

    # TODO, currently we do not split the dataset, we index the whole dataset, but load_dataset need to specify train/test, so we store everything in train
    indexer.index(es, dataset=dataset["train"])
    # indexer.index(dataset=dataset['train'])


if __name__ == "__main__":
    main()
