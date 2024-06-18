import json

import hydra
from omegaconf import OmegaConf

from zoning.class_types import OCREntities, ZoningConfig
from zoning.ocr.textract import TextractExtractor


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """Main function to run the ocr process based on the provided
    configuration.

    Configs:
        - global_config: GlobalConfig.
        - ocr_config: OCRConfig

    OCR Input File Format:
        OCREntities object, which also specifies the output directory of the OCR results.

    OCR Output File Format:
        OCREntities
    """

    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    ocr_config = ZoningConfig(config=config).ocr_config

    # Read the input data
    ocr_entities = OCREntities(
        target_names_file=global_config.target_names_file,
        pdf_dir=global_config.pdf_dir,
        ocr_results_dir=global_config.ocr_results_dir,
    )

    # Extract the data
    match ocr_config.method:
        case "textract":
            extractor = TextractExtractor(ocr_config)
        case _:
            raise ValueError(f"Extractor {ocr_config.method} not implemented")

    extractor.extract(ocr_entities)

    # Write the output data, data type is OCREntities
    with open(global_config.data_flow_ocr_file, "w") as f:
        json.dump(ocr_entities.model_dump_json(), f)


if __name__ == "__main__":
    main()
