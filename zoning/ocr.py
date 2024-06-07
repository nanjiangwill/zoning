import hydra
from ocr import ExtractionEntities, TextractExtractor
from omegaconf import DictConfig, OmegaConf
from utils import publish_dataset


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    """Main function to run the extraction process based on the provided
    configuration.

    Args:
        config (DictConfig): Configuration object specified in ../config/<config_name>.yaml

    Extraction Input File Format:
        The input should be the names of the targets to be extracted and their pdfs files

    Output File Format:
        JSON files, each containing a list of dictionaries, where each dictionary represents extracted result from a page.
    """
    OmegaConf.resolve(config)
    match config.extract.name:
        case "textract":
            extractor = TextractExtractor(config)
        case _:
            raise ValueError(f"Extractor {config.extract.name} not implemented")

    extraction_targets = ExtractionEntities(config)

    extractor.extract(extraction_targets)

    if config.extract.hf_dataset.publish_dataset:
        publish_dataset(extraction_targets, config)


if __name__ == "__main__":
    main()
