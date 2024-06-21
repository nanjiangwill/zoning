import hydra
from omegaconf import OmegaConf

from zoning.class_types import FormatOCR, ZoningConfig
from zoning.index.keyword_indexer import KeywordIndexer
from zoning.utils import process


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """Main function to run the index process based on the provided
    configuration.

    Configs:
        - global_config: GlobalConfig.
        - index_config: IndexConfig

    Input File Format:
        FormatOCR object
        config.format_ocr_dir

    Output File Format:
        None
    """
    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    index_config = ZoningConfig(config=config).index_config

    # Load the indexer
    match index_config.method:
        case "keyword":
            indexer = KeywordIndexer(index_config)
        case "embedding":
            raise NotImplementedError  # indexer = EmbeddingIndexer(config)
        case _:
            raise ValueError(f"Extractor {index_config.name} not implemented")

    process(
        global_config.target_town_file,
        global_config.format_ocr_dir,
        global_config.index_dir,
        indexer.index,
        converter=lambda x: FormatOCR.model_construct(**x),
        mode="index",
        output=False,
    )


if __name__ == "__main__":
    main()
