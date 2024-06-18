import json
import random

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.contrib.concurrent import thread_map

from zoning.class_types import (
    AllEvaluationResults,
    EvaluationDatum,
    EvaluationDatumResult,
    Place,
)
from zoning.search import KeywordSearcher, Searcher
from zoning.utils import if_town_in_evaluation_dataset


def testing_search(
    evaluation_datum: EvaluationDatum, searcher: Searcher
) -> EvaluationDatumResult | None:
    try:
        search_results = searcher.search(evaluation_datum)
        if not search_results:
            return None

        # sort search results by score
        search_results = sorted(search_results, key=lambda x: x.score, reverse=True)

        return EvaluationDatumResult(
            place=evaluation_datum.place,
            eval_term=evaluation_datum.eval_term,
            search_results=search_results,
            llm_inference_results=[],
        )

    except Exception as e:
        print(f"Error processing {evaluation_datum.place} {evaluation_datum.eval_term}")
        print(e)
        return None


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    """Main function to test the search process.

    Args:
        config (DictConfig): Configuration object specified in ../config/<config_name>.yaml

    Input File Format:
        a list of EvaluationDatum objects.

    Output File Format:
        list of SearchResult objects.
    """

    OmegaConf.resolve(config)

    # load searcher
    match config.search.method:
        case "keyword":
            searcher = KeywordSearcher(config)
        case "embedding":
            raise NotImplementedError("Embedding searcher is not implemented yet")
        case _:
            raise ValueError(f"Search method {config.search.method} is not supported")
    eval_terms = config.eval_terms

    # load evaluation data from ground truth
    with open(config.ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    evaluation_dataset_by_term = {}
    for gt_data in ground_truth:
        # check if the town is already in the evaluation dataset
        if not if_town_in_evaluation_dataset(config.dataset_dir, gt_data["town"]):
            continue
        place = Place(
            town=gt_data["town"],
            district_full_name=gt_data["district"],
            district_short_name=gt_data["district_abb"],
        )
        for eval_term in eval_terms:
            evaluation_datum = EvaluationDatum(
                place=place,
                eval_term=eval_term,
                is_district_fuzzy=searcher.is_district_fuzzy,
                is_eval_term_fuzzy=searcher.is_eval_term_fuzzy,
                thesaurus_file=searcher.thesaurus_file,
            )
            evaluation_dataset_by_term.setdefault(eval_term, []).append(
                evaluation_datum
            )

    # the target is to get evaluation_dataset in correct type
    if config.random_seed and (
        config.test_size_per_term or config.test_percentage_by_term
    ):
        random.seed(config.random_seed)
        for eval_term, evaluation_dataset in evaluation_dataset_by_term.items():
            random.shuffle(evaluation_dataset)
            evaluation_dataset = evaluation_dataset[: config.test_size_per_term]
            evaluation_dataset_by_term[eval_term] = evaluation_dataset
    else:
        raise ValueError("random_seed and test_size must be provided to save cost")

    evaluation_datum_result_list = thread_map(
        lambda x: testing_search(x, searcher),
        [
            evaluation_datum
            for evaluation_dataset in evaluation_dataset_by_term.values()
            for evaluation_datum in evaluation_dataset
        ],
        max_workers=config.num_workers,
    )

    evaluation_datum_result_list = [
        x for x in evaluation_datum_result_list if x is not None
    ]
    all_evaluation_results = AllEvaluationResults(
        all_evaluation_results=evaluation_datum_result_list
    )

    all_evaluation_results.save_to(config.testing.result_output_dir)


if __name__ == "__main__":
    main()
