import hydra
from index import IndexEntities, KeywordIndexer
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    """Main function to run the index process based on the provided
    configuration.

    Args:
        config (DictConfig): Configuration object specified in ../config/<config_name>.yaml

    Index Input File Format:
        The input should be a list of IndexEntity objects, where each object contains the index data for a page.
    """
    OmegaConf.resolve(config)
    match config.index.method:
        case "keyword":
            indexer = KeywordIndexer(config)
        case "embedding":
            raise NotImplementedError  # indexer = EmbeddingIndexer(config)
        case _:
            raise ValueError(f"Extractor {config.extract.name} not implemented")

    index_entities = IndexEntities(config)

    # TODO, currently we do not split the dataset, we index the whole dataset, but load_dataset need to specify train/test, so we store everything in train
    indexer.index(index_entities)


if __name__ == "__main__":
    main()
