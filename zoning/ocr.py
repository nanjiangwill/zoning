import json
import os

import hydra
from ocr import TextractExtractor
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    """
    The OCR step takes in a json file of pointers to town pdf and outputs
    json.
    """


    OmegaConf.resolve(config)
    match config.extract.name:
        case "textract":
            extractor = TextractExtractor(config)
        case _:
            raise ValueError(f"Extractor {config.extract.name} not implemented")

    state_data_path = os.path.join(config.data_output_dir, config.target_state)
    state_all_towns_names_path = os.path.join(state_data_path, "all_towns_names.json")

    with open(state_all_towns_names_path, "r") as f:
        state_all_towns_names = json.load(f)

    extractor.extract(state_all_towns_names)
    extractor.post_extract(state_all_towns_names)


if __name__ == "__main__":
    main()
