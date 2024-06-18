import json

import hydra
from omegaconf import OmegaConf

from zoning.class_types import Place, SearchQueries, SearchQuery, ZoningConfig
from zoning.search.keyword_searcher import KeywordSearcher


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

    # Construct the search query and use it to query the indexer
    # load search query data from ground truth
    with open(global_config.ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    search_queries = []
    for gt_data in ground_truth:
        place = Place(
            town=gt_data["town"],
            district_full_name=gt_data["district"],
            district_short_name=gt_data["district_abb"],
        )
        for eval_term in global_config.eval_terms:
            search_query = SearchQuery(place=place, eval_term=eval_term)
            search_queries.append(search_query)

    search_queries = SearchQueries(search_queries=search_queries)

    # Load the searcher
    match search_config.method:
        case "keyword":
            searcher = KeywordSearcher(search_config)
        case "embedding":
            raise NotImplementedError("Embedding searcher is not implemented yet")
        case _:
            raise ValueError(f"Search method {config.search.method} is not supported")

    search_results = searcher.search(search_queries)

    # Write the output data, data type is SearchResults
    with open(global_config.data_flow_search_file, "w") as f:
        json.dump(search_results.model_dump_json(), f)


if __name__ == "__main__":
    main()
