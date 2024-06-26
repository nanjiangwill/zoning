import json
import os

import hydra
from omegaconf import OmegaConf

from zoning.class_types import (
    DistrictEvalResult,
    NormalizedLLMInferenceResult,
    ZoningConfig,
)
from zoning.utils import process, flatten


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
        ground_truth_page_int = [int(ground_truth_page)] if "," not in ground_truth_page else [int(x) for x in ground_truth_page.split(",")]
        search_range = flatten([lo.llm_output.search_page_range for lo in d.normalized_llm_outputs])
        page_in_range = any(i in search_range for i in ground_truth_page_int)
        answer_correct = False
        for o in d.normalized_llm_outputs:
            if o.normalized_answer and (
                ground_truth in o.normalized_answer
                or ground_truth_orig in o.normalized_answer
            ):
                answer_correct = True
                break
            
    a = DistrictEvalResult(
        place=d.place,
        eval_term=d.eval_term,
        search_result=d.search_result,
        input_prompts=d.input_prompts,
        normalized_llm_outputs=d.normalized_llm_outputs,
        ground_truth=ground_truth,
        ground_truth_orig=ground_truth_orig,
        ground_truth_page=ground_truth_page,
        answer_correct=answer_correct,
        page_in_range=page_in_range,
    )
    return DistrictEvalResult(
        place=d.place,
        eval_term=d.eval_term,
        search_result=d.search_result,
        input_prompts=d.input_prompts,
        normalized_llm_outputs=d.normalized_llm_outputs,
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

    Input File Format:
        NormalizedLLMInferenceResult
        config.normalization_dir

    Output File Format:
        DistrictEvalResult
        config.eval_dir
    """
    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    # eval_config = ZoningConfig(config=config).eval_config

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
            DistrictEvalResult.model_construct(**json.load(open(f)))
            for f in eval_term_files
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


if __name__ == "__main__":
    main()
