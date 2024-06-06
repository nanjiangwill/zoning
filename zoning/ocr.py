import os

import hydra
from class_types import ExtractionTargetCollection
from ocr import TextractExtractor
from omegaconf import DictConfig, OmegaConf
from utils import publish_dataset


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    """
    Input data format: pdf
    Output data format: json file. Containing a list of json objects, each object contains the extracted text from a page.
    """
    OmegaConf.resolve(config)
    match config.extract.name:
        case "textract":
            extractor = TextractExtractor(config)
        case _:
            raise ValueError(f"Extractor {config.extract.name} not implemented")

    extraction_target = ExtractionTargetCollection(config)

    extractor.extract(extraction_target)
    if config.extract.hf_dataset.publish_dataset:
        publish_dataset(extraction_target, config)
        from datasets import load_dataset

        hf_dataset_files = [
            os.path.join(extraction_target.dataset_dir, file)
            for file in os.listdir(extraction_target.dataset_dir)
        ]

        page_dataset = load_dataset("json", data_files=hf_dataset_files)

        if config.extract.hf_dataset.publish_dataset:
            page_dataset.push_to_hub(
                config.extract.hf_dataset.name,
                private=config.extract.hf_dataset.private,
            )


if __name__ == "__main__":
    main()
