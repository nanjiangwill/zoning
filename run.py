import hydra
import wandb 
import os
from omegaconf import OmegaConf
# Import the main functions from each module
from zoning.ocr.__main__ import main as ocr_main
from zoning.format_ocr.__main__ import main as format_ocr_main
from zoning.index.__main__ import main as index_main
from zoning.search.__main__ import main as search_main
from zoning.prompt.__main__ import main as prompt_main
from zoning.llm.__main__ import main as llm_main
from zoning.normalization.__main__ import main as normalization_main
from zoning.eval.__main__ import main as eval_main
from zoning.class_types import ZoningConfig

@hydra.main(version_base=None, config_path="config", config_name="base")
def run_all(config: ZoningConfig):
    
    wandb.init(project="zoning", name=config["global_config"]['experiment_name'], config=OmegaConf.to_object(config))
    
    # print("Running OCR module:")
    # ocr_main(config)

    print("Running Format OCR module:")
    format_ocr_main(config)
    
    print("Running Index module:")
    index_main(config)
    
    print("Running Search module:")
    search_main(config)
    
    print("Running Prompt module:")
    prompt_main(config)
    
    print("Running LLM module:")
    llm_main(config["global_config"]['experiment_name'])
    
    print("Running Normalization module:")
    normalization_main(config)
    
    print("Running Evaluation module:")
    eval_main(config)

if __name__ == "__main__":
    run_all()