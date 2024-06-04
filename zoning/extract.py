import hydra
from omegaconf import DictConfig, OmegaConf
from .extract import *


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    match config.extract.name:
        case "textract":
            extractor = TextractExtractor(config)
        case _:
            raise ValueError(f"Extractor {config.extract.name} not implemented")

    extractor.extract()
    extractor.post_extract()


if __name__ == "__main__":
    main()
