import hydra
from omegaconf import OmegaConf

from zoning.class_types import ZoningConfig
from zoning.ocr.textract import TextractExtractor


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """Converts PDF files to textract json format.

    Configs:
        - global_config: GlobalConfig.
        - ocr_config: OCRConfig

    Input File Format:
        PDF files
        global_config.pdf_dir

    Output File Format:
        Textract serialized objects
        global_config.ocr_results_dir
    """

    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    ocr_config = ZoningConfig(config=config).ocr_config

    # Extract the data
    match ocr_config.method:
        case "textract":
            ocr = TextractExtractor(ocr_config)
        case _:
            raise ValueError(f"Extractor {ocr_config.method} not implemented")

    num_targets = ocr.process_files_and_write_output(
        global_config.target_town_file, global_config.pdf_dir, global_config.ocr_dir
    )

    print()
    print(f"Stage OCR Completed. Processed {num_targets} targets. Data saved to {global_config.ocr_dir}")



if __name__ == "__main__":
    main()
