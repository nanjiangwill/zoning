import json
from functools import partial
from typing import Dict, List

import hydra
from omegaconf import OmegaConf
from tqdm.contrib.concurrent import thread_map

from zoning.class_types import (
    EvalMetricByTerm,
    EvalQuery,
    LLMInferenceResult,
    LLMInferenceResults,
    ZoningConfig,
)
from zoning.utils import semantic_comparison


def eval_term_metric(
    eval_term: str,
    ground_truth: List[Dict[str, str]],
    llm_inference_results: List[LLMInferenceResult],
) -> EvalMetricByTerm:
    eval_queries = []
    for d in llm_inference_results:
        d_ground_truth_info = list(
            filter(
                lambda x: x["town"] == d.place.town
                and x["district"] == d.place.district_full_name
                and x["district_abb"] == d.place.district_short_name,
                ground_truth,
            )
        )

        # if there is no ground truth for this evaluation data, skip
        if d_ground_truth_info is None:
            continue

        # there show be only one matching for each evaluation data
        d_ground_truth_info = d_ground_truth_info[0]

        d_ground_truth = d_ground_truth_info[f"{eval_term}_gt"]
        d_ground_truth_orig = d_ground_truth_info[f"{eval_term}_gt_orig"]
        d_ground_truth_page = d_ground_truth_info[f"{eval_term}_page_gt"]

        eval_query = EvalQuery(
            place=d.place,
            eval_term=eval_term,
            search_result=d.search_result,
            llm_inference_result=d,
            ground_truth=d_ground_truth,
            ground_truth_orig=d_ground_truth_orig,
            ground_truth_page=d_ground_truth_page,
        )
        eval_queries.append(eval_query)

    # TODOL use tool to do evaluation

    # Calculate metrics
    # WIP
    # can add more metrics
    answer_tp = answer_fp = answer_fn = 0
    page_tp = page_fp = page_fn = 0
    correct_search_and_llm_inference_pair_list = []
    for eval_query in eval_queries:
        answer_flag = False
        page_flag = False
        correct_pair = None
        is_in_entire_search_page_range = (
            eval_query.ground_truth_page in eval_query.entire_search_results_page_range
        )
        if eval_query.ground_truth is None:
            continue
        for search_result, llm_inference_result in zip(
            eval_query.search_results,
            eval_query.llm_inference_results,
        ):

            lambda x: semantic_comparison(
                x["actual"], x["expected_extended"]
            ) or semantic_comparison(x["actual"], x["expected"])
            if (
                llm_inference_result.answer is not None
                and (
                    semantic_comparison(
                        llm_inference_result.answer,
                        eval_query.ground_truth,
                    )
                    or semantic_comparison(
                        llm_inference_result.answer,
                        eval_query.ground_truth_orig,
                    )
                )
                and eval_query.ground_truth_page in search_result.page_range
            ):
                answer_flag = True
                page_flag = True
                correct_pair = (search_result, llm_inference_result)
                correct_search_and_llm_inference_pair_list.append(correct_pair)
                break

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

    eval_metric_by_term = EvalMetricByTerm(
        eval_term=eval_term,
        answer_f1=2 * answer_tp / (2 * answer_tp + answer_fp + answer_fn),
        answer_precision=answer_tp / (answer_tp + answer_fp),
        answer_recall=answer_tp / (answer_tp + answer_fn),
        page_f1=2 * page_tp / (2 * page_tp + page_fp + page_fn),
        page_precision=page_tp / (page_tp + page_fp),
        page_recall=page_tp / (page_tp + page_fn),
        is_in_entire_search_page_range=is_in_entire_search_page_range,
    )

    return eval_queries, eval_metric_by_term


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """Main function to run the search and LLM inference process based on the
    provided configuration.

     Configs:
        - global_config: GlobalConfig.
        - eval_config: OCRConfig

    Input File Format:
        LLMInferenceResults

    Output File Format:
        EvalMetricByTerm objects for each evaluation term.
    """
    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    eval_config = ZoningConfig(config=config).eval_config

    # Read the input data
    with open(global_config.data_flow_llm_file, "r") as f:
        data_string = json.load(f)
        llm_inference_results = LLMInferenceResults.model_construct(
            **json.loads(data_string)
        )

    eval_term_metric_results = thread_map(
        partial(
            eval_term_metric,
            ground_truth=llm_inference_results.ground_truth,
        ),
        llm_inference_results.llm_inference_results_by_eval_term,
    )

    # Write the output data
    for eval_queries, eval_metric_by_term in eval_term_metric_results:
        print(eval_metric_by_term)
        with open(
            f"{global_config.data_flow_eval_file.replace("json", "")}/{eval_metric_by_term.eval_term}.json",
            "w",
        ) as f:
            json.dump(eval_queries.model_dump_json(), f)

        with open(
            f"{global_config.data_flow_eval_result_file.replace("json", "")}/{eval_metric_by_term.eval_term}.json",
            "w",
        ) as f:
            json.dump(eval_metric_by_term.model_dump_json(), f)


if __name__ == "__main__":
    main()
