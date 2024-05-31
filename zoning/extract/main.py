import hydra
from omegaconf import DictConfig
from extractor import *


@hydra.main(version_base=None, config_path="config/extract", config_name="base")
def main(config: DictConfig):
    match config.extract.name:
        case "textract":
            extractor = TextractExtractor(config)
        case _:
            raise ValueError(f"Extractor {config.extract.name} not implemented")

    extractor.extract()
    extractor.post_extract()


if __name__ == "__main__":
    main()
