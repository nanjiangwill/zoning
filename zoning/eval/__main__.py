import json
import os

import hydra
from omegaconf import OmegaConf

from zoning.class_types import (
    DistrictEvalResult,
    NormalizedLLMInferenceResult,
    ZoningConfig,
)
from zoning.utils import process


def eval_fn(d, gt, target) -> DistrictEvalResult:
    gt_info = list(
        filter(
            lambda x: x["town"] == d.place.town
            and x["district"] == d.place.district_full_name
            and x["district_abb"] == d.place.district_short_name,
            gt,
        )
    )
    if gt_info is None or len(gt_info) == 0:
        ground_truth = None
        ground_truth_orig = None
        ground_truth_page = None
        answer_correct = None
        page_in_range = None
    else:
        # there show be only one matching for each evaluation data
        gt_info = gt_info[0]
        ground_truth = gt_info[f"{d.eval_term}_gt"]
        ground_truth_orig = gt_info[f"{d.eval_term}_gt_orig"]
        ground_truth_page = gt_info[f"{d.eval_term}_page_gt"]
        page_in_range = ground_truth_page in d.search_result.entire_search_page_range
        answer_correct = False
        for o in d.normalized_llm_outputs:
            if (
                ground_truth in o.normalized_answer
                or ground_truth_orig in o.normalized_answer
            ):
                answer_correct = True
                break

    return DistrictEvalResult(
        place=d.place,
        eval_term=d.eval_term,
        normalized_llm_inference_result=d,
        ground_truth=ground_truth,
        ground_truth_orig=ground_truth_orig,
        ground_truth_page=ground_truth_page,
        answer_correct=answer_correct,
        page_in_range=page_in_range,
    )


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """Main function to run the evaluation process based on the provided
    configuration.

     Configs:
        - global_config: GlobalConfig.
        - eval_config: EvalConfig

    Eval Input File Format:
        NormalizedLLMInferenceResult

    Eval Output File Format:
        EvalMetricByTerm objects for each evaluation term.
    """
    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    eval_config = ZoningConfig(config=config).eval_config

    # Read the input data and read ground truth
    # llm_inference_results = LLMInferenceResults.model_construct(
    #     **json.load(open(global_config.data_flow_llm_file))
    # )
    # test_data = json.load(open(global_config.test_data_file))

    process(
        global_config.target_eval_file,
        global_config.normalization_dir,
        global_config.eval_dir,
        fn=lambda x, y: eval_fn(x, json.load(open(global_config.ground_truth_file)), y),
        converter=lambda x: NormalizedLLMInferenceResult.model_construct(**x),
    )

    # Calculate metrics
    for term in global_config.eval_terms:
        eval_term_files = [
            os.path.join(global_config.eval_dir, f)
            for f in os.listdir(global_config.eval_dir)
            if f.startswith(term)
        ]
        eval_term_data = [
            DistrictEvalResult(**json.load(open(f))) for f in eval_term_files
        ]
        accuracy = sum([1 for d in eval_term_data if d.answer_correct]) / len(
            eval_term_data
        )
        page_precision = sum([1 for d in eval_term_data if d.page_in_range]) / len(
            eval_term_data
        )

        print(f"Evaluated term: {term}")
        print(f"Accuracy: {accuracy}")
        print(f"Page Precision: {page_precision}")

    # for term in global_config.eval_terms:
    #     accuracy, page_precision = eval_term(
    #         term,
    #         global_config.normalization_dir,
    #         global_config.ground_truth_file,
    #         converter=lambda x: NormalizedLLMInferenceResult.model_construct(**x),
    #     )
    #     print(f"Evaluated term: {term}")
    #     print(f"Accuracy: {accuracy}")
    #     print(f"Page Precision: {page_precision}")

    # Run the evaluation
    # eval_term_metric_results = [
    #     evaluator.eval_term_metric(
    #         eval_term,
    #         test_data,
    #         llm_inference_results.llm_inference_results_by_eval_term[eval_term],
    #     )
    #     for eval_term in llm_inference_results.llm_inference_results_by_eval_term
    # ]

    # # Write the output data
    # with open(global_config.data_flow_eval_file, "w") as f:
    #     json.dump(
    #         [eval_queries.model_dump() for eval_queries, _ in eval_term_metric_results],
    #         f,
    #     )
    # with open(global_config.data_flow_eval_result_file, "w") as f:
    #     json.dump(
    #         [
    #             eval_metric_by_term.model_dump()
    #             for _, eval_metric_by_term in eval_term_metric_results
    #         ],
    #         f,
    #     )


if __name__ == "__main__":
    main()
