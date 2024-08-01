import json

import hydra
from omegaconf import OmegaConf

from zoning.class_types import SearchQuery, ZoningConfig
from zoning.search.keyword_searcher import KeywordSearcher
from zoning.utils import process


def preprocess_search_target(town_file, district_file, eval_terms, output_file):
    all_town = json.load(open(town_file))
    districts = json.load(open(district_file))
    search_targets = []
    for district in districts:
        if district.split("__")[0] not in all_town:
            continue
        town, district_short_name, district_full_name = district.split("__")
        for term in eval_terms:
            search_targets.append(
                f"{term}__{town}__{district_short_name}__{district_full_name}"
            )
    with open(output_file, "w") as f:
        json.dump(sorted(search_targets), f, indent=4)


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """Main function to run the search process based on the provided
    configuration.

    Configs:
        - global_config: GlobalConfig.
        - search_config: SearchConfig

    Input File Format:
        None
        Need to construct the target with `preprocess_search_target`

    Output File Format:
        SearchResult
        config.search_dir
    """
    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    search_config = ZoningConfig(config=config).search_config

    # Load the searcher
    match search_config.method:
        case "keyword":
            searcher = KeywordSearcher(search_config)
        case "embedding":
            raise NotImplementedError("Embedding searcher is not implemented yet")
        case _:
            raise ValueError(f"Search method {search_config.method} is not supported")

    # Construct the entire search query with all possible eval terms

    if search_config.preprocess_search_target:
        preprocess_search_target(
            global_config.target_town_file,
            global_config.target_district_file,
            global_config.eval_terms,
            global_config.target_eval_file,
        )

    process(
        global_config.target_eval_file,
        None,
        global_config.search_dir,
        searcher.search,
        read_fn=lambda x, y: x,
        converter=lambda x: SearchQuery(raw_query_str=x),
    )


if __name__ == "__main__":
    main()
