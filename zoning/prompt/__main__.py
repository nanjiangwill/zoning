import json

import hydra
from omegaconf import OmegaConf

from zoning.class_types import SearchResult, ZoningConfig
from zoning.prompt.prompt_generator import PromptGenerator
from zoning.utils import process


def preprocess_search_target(town_file, district_file, eval_terms, output_file):
    town = json.load(open(town_file))
    districts = json.load(open(district_file))
    search_targets = []
    for district in districts:
        if district.split("__")[0] not in town:
            continue
        town, district_short_name, district_full_name = district.split("__")
        for term in eval_terms:
            search_targets.append(
                f"{term}__{town}__{district_short_name}__{district_full_name}"
            )
    with open(output_file, "w") as f:
        json.dump(search_targets, f, indent=4)
    return search_targets


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

    # # Construct the entire search query with all possible eval terms
    # search_queries = SearchQueries(query_file=global_config.test_data_file)

    # # filter the search queries based on the eval terms we need and the test size
    # search_queries.get_test_data_search_queries(
    #     global_config.eval_terms,
    #     global_config.random_seed,
    #     global_config.test_size_per_term,
    # )

    # Load the searcher
    prompt_generator = PromptGenerator(prompt_config)


    process(
        global_config.target_eval_file,
        global_config.search_dir,
        global_config.prompt_dir,
        prompt_generator.generate_prompt,
        converter=lambda x: SearchResult.model_construct(**x),
    )

if __name__ == "__main__":
    main()
