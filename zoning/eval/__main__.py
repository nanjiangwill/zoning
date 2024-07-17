import json
import os

import hydra
from omegaconf import OmegaConf

from zoning.class_types import (
    DistrictEvalResult,
    NormalizedLLMInferenceResult,
    ZoningConfig,
)
from zoning.utils import flatten, process


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
        if ground_truth_page is not None:
            ground_truth_page_int = (
                [int(ground_truth_page)]
                if "," not in ground_truth_page
                else [int(x) for x in ground_truth_page.split(",")]
            )
        else:
            ground_truth_page_int = []

        search_ranges = [lo.llm_output.search_page_range for lo in d.normalized_llm_outputs]
        page_in_range = []
        
        if len(ground_truth_page_int) == 0:
            page_in_range = [False for _ in search_ranges]
        else:
            for sr in search_ranges:
                if any(i in sr for i in ground_truth_page_int):
                    page_in_range.append(True)
                else:
                    page_in_range.append(False)
                    
        answer_correct = []
        
        for o in d.normalized_llm_outputs:
            if o.llm_output.raw_model_response is None:
                continue
            if ground_truth is None and ground_truth_orig is None:
                if o.normalized_answer is None:
                    answer_correct.append(True)
                    continue
            else:
                if o.normalized_answer and (
                    ground_truth in o.normalized_answer
                    or ground_truth_orig in o.normalized_answer
                ):
                    answer_correct.append(True)
                    continue
            answer_correct.append(False)
                

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
        page_in_range=page_in_range
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
        
        all_accuracy_results = [d.answer_correct for d in eval_term_data]
        
        best_accuracy = sum([1 for d in all_accuracy_results if any(d)])/ len(all_accuracy_results)
        avg_accuracy = sum([sum([1 for d in a if d]) for a in all_accuracy_results]) / sum([len(a) for a in all_accuracy_results])
        
        all_page_results = [d.page_in_range for d in eval_term_data]
        
        best_page_in_range = sum([1 for d in all_page_results if any(d)])/ len(all_page_results)
        avg_page_in_range = sum([sum([1 for d in a if d]) for a in all_page_results]) / sum([len(a) for a in all_page_results])
        

        print("=============================================")
        print(f"Evaluated term: {term}")
        print(f"Best Accuracy: {best_accuracy}")
        print(f"Avg Accuracy: {avg_accuracy}")
        print("\n")
        print(f"Best Page Accuracy: {best_page_in_range}")
        print(f"Avg Page Accuracy: {avg_page_in_range}")
        print("=============================================")
        print("\n")


if __name__ == "__main__":
    main()
