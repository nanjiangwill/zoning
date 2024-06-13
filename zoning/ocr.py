import hydra
from ocr import TextractExtractor
from omegaconf import DictConfig, OmegaConf

from zoning.class_types import ExtractionEntities
from zoning.utils import publish_dataset


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    """Main function to run the extraction process based on the provided
    configuration.

    Args:
        config (DictConfig): Configuration object specified in ../config/<config_name>.yaml

    Extraction Input Format:
        List of ExtractionEntities

    Output File Format:
        There is no data structure for the output, as it is dependent on the extractor used.
        The output is JSON files, each containing a list of dictionaries, where each dictionary represents extracted result from a page.
    """
    OmegaConf.resolve(config)
    match config.extract.name:
        case "textract":
            extractor = TextractExtractor(config)
        case _:
            raise ValueError(f"Extractor {config.extract.name} not implemented")

    extraction_targets = ExtractionEntities(
        pdf_dir=config.pdf_dir,
        ocr_result_dir=config.ocr_result_dir,
        dataset_dir=config.dataset_dir,
        target_names_file=config.target_names_file,
    )

    extractor.extract(extraction_targets)

    if config.extract.hf_dataset.publish_dataset:
        publish_dataset(extraction_targets, config)


if __name__ == "__main__":
    main()
