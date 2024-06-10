import json
import os
from functools import partial

import hydra
import polars as pl
from class_types import EvaluationDatumResult, EvaluationMetricByTerm
from omegaconf import DictConfig, OmegaConf
from tqdm.contrib.concurrent import process_map


def eval_term_metrics(eval_term: str, eval_result_dir: str, ground_truth: pl.DataFrame):
    eval_term_result_file = os.path.join(eval_result_dir, f"{eval_term}.json")
    evaluation_data = []

    # Load evaluation data
    with open(eval_term_result_file, "r") as f:
        data = json.load(f)
        for d in data:
            evaluation_datum_result = EvaluationDatumResult(**json.loads(d))

            # Load corresponding ground truth
            evaluation_datum_result_ground_truth = ground_truth.filter(
                (pl.col("town") == evaluation_datum_result.place.town)
                & (
                    pl.col("district")
                    == evaluation_datum_result.place.district_full_name
                )
                & (
                    pl.col("district_abb")
                    == evaluation_datum_result.place.district_short_name
                )
            )[f"{eval_term}_gt", f"{eval_term}_page_gt"]

            evaluation_datum_result.ground_truth = evaluation_datum_result_ground_truth[
                f"{eval_term}_gt"
            ].item()
            evaluation_datum_result.ground_truth_page = (
                evaluation_datum_result_ground_truth[f"{eval_term}_page_gt"].item()
            )

            evaluation_data.append(evaluation_datum_result)

    print(f"Loaded {len(evaluation_data)} evaluation data for evaluating {eval_term}")

    # Calculate metrics
    # WIP
    answer_tp = 0
    answer_fp = 0
    answer_fn = 0
    page_tp = 0
    page_fp = 0
    page_fn = 0
    for evaluation_datum_result in evaluation_data:
        answer_flag = False
        page_flag = False
        is_in_entire_search_page_range = (
            evaluation_datum_result.ground_truth_page
            in evaluation_datum_result.entire_search_results_page_range
        )
        for search_result, llm_inference_result in zip(
            evaluation_datum_result.search_results,
            evaluation_datum_result.llm_inference_results,
        ):
            if (
                llm_inference_result.answer is not None
                and evaluation_datum_result.ground_truth in llm_inference_result.answer
                and evaluation_datum_result.ground_truth_page
                in search_result.page_range
            ):
                answer_flag = True
                page_flag = True

        if answer_flag:
            answer_tp += 1
        else:
            answer_fp += 1
            answer_fn += 1
        if page_flag:
            page_tp += 1
        else:
            page_fp += 1
            page_fn += 1

    evaluation_by_term = EvaluationMetricByTerm(
        eval_term=eval_term,
        answer_f1=2 * answer_tp / (2 * answer_tp + answer_fp + answer_fn),
        answer_precision=answer_tp / (answer_tp + answer_fp),
        answer_recall=answer_tp / (answer_tp + answer_fn),
        page_f1=2 * page_tp / (2 * page_tp + page_fp + page_fn),
        page_precision=page_tp / (page_tp + page_fp),
        page_recall=page_tp / (page_tp + page_fn),
        is_in_entire_search_page_range=is_in_entire_search_page_range,
    )

    with open(os.path.join(eval_result_dir, f"{eval_term}_metrics.json"), "w") as f:
        json.dump(evaluation_by_term.model_dump(), f)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    """Main function to run the search and LLM inference process based on the
    provided configuration.

    Args:
        config (DictConfig): Configuration object specified in ../config/<config_name>.yaml

    Input File Format:
        The input should be the json serialized list of EvaluationDatumResult objects.

    Output File Format:
        EvaluationMetricByTerm objects for each evaluation term. This will be serialized to a json file.
        {
            eval_term:
            answer_f1:
            answer_precision:
            answer_recall:
            page_f1:
            page_precision:
            page_recall:
            is_in_entire_search_page_range:
        }
    """
    OmegaConf.resolve(config)

    eval_terms = config.eval_terms
    ground_truth = pl.read_csv(
        config.ground_truth_file,
        schema_overrides={
            **{f"{tc}_gt": pl.Utf8 for tc in eval_terms},
            **{f"{tc}_page_gt": pl.Utf8 for tc in eval_terms},
        },
    )
    process_map(
        partial(
            eval_term_metrics,
            eval_result_dir=config.result_output_dir,
            ground_truth=ground_truth,
        ),
        eval_terms,
    )


if __name__ == "__main__":
    main()
