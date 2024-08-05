import glob
import json

import hydra
from omegaconf import OmegaConf

from zoning.class_types import (
    DistrictExtractionResult,
    DistrictExtractionVerificationResult,
    FormatOCR,
    PageEmbeddingResult,
    ZoningConfig,
)
from zoning.district_extraction.district_extractor import DistrictExtractor
from zoning.utils import process


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """Get the embeddings of the pages and extract the districts based on the
    provided configuration.

    Configs:
        - global_config: GlobalConfig.
        - page_embedding_config: PageEmbeddingConfig

    Input File Format:
        FormatOCR
        config.format_ocr_dir

    Output File Format:
        PageEmbeddingResult
        config.page_embedding_dir

        DistrictExtractionResult
        config.district_extraction_dir
    """

    # Parse the config
    config = ZoningConfig(config=OmegaConf.to_object(config))
    global_config = config.global_config
    district_extraction_config = config.district_extraction_config
    district_extractor = DistrictExtractor(
        district_extraction_config=district_extraction_config
    )

    # Construct the input data
    if district_extraction_config.run_district_extraction:
        process(
            global_config.target_town_file,
            global_config.format_ocr_dir,
            global_config.page_embedding_dir,
            district_extractor.page_embedding,
            converter=lambda x: FormatOCR.model_construct(**x),
        )

        process(
            global_config.target_town_file,
            global_config.page_embedding_dir,
            global_config.district_extraction_dir,
            district_extractor.district_extraction,
            converter=lambda x: PageEmbeddingResult.model_construct(**x),
        )

        process(
            global_config.target_town_file,
            global_config.district_extraction_dir,
            global_config.district_extraction_verification_dir,
            district_extractor.district_extraction_verification,
            converter=lambda x: DistrictExtractionResult.model_construct(**x),
        )

        all_districts_files = glob.glob(
            f"{global_config.district_extraction_verification_dir}/*.json"
        )
        districts = []
        for district_file in all_districts_files:
            verified_districts = DistrictExtractionVerificationResult.model_construct(
                **json.load(open(district_file))
            )
            districts.extend(verified_districts.valid_districts)

        districts = sorted(list(set(districts)))
        with open(district_extraction_config.target_districts_file, "w") as f:
            json.dump(districts, f, indent=4)


if __name__ == "__main__":
    main()
