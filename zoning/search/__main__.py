import json

import hydra
from omegaconf import OmegaConf

from zoning.class_types import SearchQuery, ZoningConfig
from zoning.search.keyword_searcher import KeywordSearcher
from zoning.utils import process


def preprocess_search_target(district_file, eval_terms, output_file):
    districts = json.load(open(district_file))
    search_targets = []
    for district in districts:
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

    Search Input File Format:
        None
        Need to construct the SearchQuery object

    Search Output File Format:
        SearchResults object
    """
    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    search_config = ZoningConfig(config=config).search_config

    # # Construct the entire search query with all possible eval terms
    # search_queries = SearchQueries(query_file=global_config.test_data_file)

    # # filter the search queries based on the eval terms we need and the test size
    # search_queries.get_test_data_search_queries(
    #     global_config.eval_terms,
    #     global_config.random_seed,
    #     global_config.test_size_per_term,
    # )

    # Load the searcher
    match search_config.method:
        case "keyword":
            searcher = KeywordSearcher(search_config)
        case "embedding":
            raise NotImplementedError("Embedding searcher is not implemented yet")
        case _:
            raise ValueError(f"Search method {search_config.method} is not supported")

    # search_results = searcher.search(search_queries)
    preprocess_search_target(
        global_config.target_district_file,
        global_config.eval_terms,
        global_config.target_eval_file,
    )

    process(
        global_config.target_eval_file,
        None,
        global_config.search_dir,
        searcher.search,
        converter=lambda x: SearchQuery(raw_query_str=x),
    )

    # Write the output data, data type is SearchResults
    # with open(global_config.data_flow_search_file, "w") as f:
    #     json.dump(search_results.model_dump(), f)


if __name__ == "__main__":
    main()
