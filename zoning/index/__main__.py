import json

import hydra
from omegaconf import OmegaConf

from zoning.class_types import OCREntities, ZoningConfig
from zoning.index.keyword_indexer import KeywordIndexer


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """Main function to run the index process based on the provided
    configuration.

    Configs:
        - global_config: GlobalConfig.
        - index_config: IndexConfig

    Index Input File Format:
        OCREntities object

    Index Output File Format:
        IndexEntities object
    """
    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    index_config = ZoningConfig(config=config).index_config

    # Read the input data
    ocr_entities = OCREntities.model_construct(
        **json.load(open(global_config.data_flow_ocr_file))
    )

    # Load the indexer
    match index_config.method:
        case "keyword":
            indexer = KeywordIndexer(index_config)
        case "embedding":
            raise NotImplementedError  # indexer = EmbeddingIndexer(config)
        case _:
            raise ValueError(f"Extractor {index_config.name} not implemented")

    index_entities = indexer.index(ocr_entities)

    # Write the output data, data type is IndexEntities
    # Since we index the data to ES, we just store the index_entities for testing purpose
    with open(global_config.data_flow_index_file, "w") as f:
        json.dump(index_entities.model_dump(), f)


if __name__ == "__main__":
    main()
