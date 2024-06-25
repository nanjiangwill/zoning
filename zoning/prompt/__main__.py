import hydra
from omegaconf import OmegaConf

from zoning.class_types import SearchResult, ZoningConfig
from zoning.prompt.prompt_generator import PromptGenerator
from zoning.utils import process


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """Main function to run the search process based on the provided
    configuration.

    Configs:
        - global_config: GlobalConfig.
        - search_config: SearchConfig

    Input File Format:
        SearchResult
        config.search_dir

    Output File Format:
        PromptResult
        config.prompt_dir
    """
    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    prompt_config = ZoningConfig(config=config).prompt_config

    # Load the searcher
    prompt_generator = PromptGenerator(prompt_config)

    num_targets = process(
        global_config.target_eval_file,
        global_config.search_dir,
        global_config.prompt_dir,
        prompt_generator.generate_prompt,
        converter=lambda x: SearchResult.model_construct(**x),
    )

    print()
    print(f"Stage Prompt Completed. Processed {num_targets} targets. Data saved to {global_config.prompt_dir}.")



if __name__ == "__main__":
    main()
