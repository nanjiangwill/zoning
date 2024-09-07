import glob
import json
import os

import hydra
from omegaconf import OmegaConf

from zoning.class_types import EvalResult, NormalizedLLMInferenceResult, ZoningConfig
from zoning.utils import process


def eval_fn(d: NormalizedLLMInferenceResult, gt, experiment_dir, target) -> EvalResult:
    if gt is None:
        gt_info = None
    else:
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

    search_file = glob.glob(f"{experiment_dir}/search/*{d.eval_term}__{d.place}.json")
    assert len(search_file) == 1
    search_result = json.load(open(search_file[0]))
    search_ranges = search_result["entire_search_page_range"]

    page_in_range = None

    if len(ground_truth_page_int) == 0:
        page_in_range = False
    else:
        if any(i in search_ranges for i in ground_truth_page_int):
            page_in_range = True
        else:
            page_in_range = False

    answer_correct = None

    assert len(d.normalized_llm_outputs) == 1

    o = d.normalized_llm_outputs[0]
    if ground_truth is None and ground_truth_orig is None:
        if o.normalized_answer is None:
            answer_correct = True
        else:
            answer_correct = False
    else:
        if o.normalized_answer and (
            ground_truth in o.normalized_answer
            or ground_truth_orig in o.normalized_answer
        ):
            answer_correct = True
        else:
            answer_correct = False

    return EvalResult(
        place=d.place,
        eval_term=d.eval_term,
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
        EvalResult
        config.eval_dir
    """
    # Parse the config
    config = ZoningConfig(config=OmegaConf.to_object(config))
    global_config = config.global_config
    # eval_config = config.eval_config

    process(
        global_config.target_eval_file,
        global_config.normalization_dir,
        global_config.eval_dir,
        fn=lambda x, y: eval_fn(
            x,
            (
                json.load(open(global_config.ground_truth_file))
                if os.path.exists(global_config.ground_truth_file)
                else None
            ),
            global_config.experiment_dir,
            y,
        ),
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
            EvalResult.model_construct(**json.load(open(f))) for f in eval_term_files
        ]

        all_accuracy_results = [d.answer_correct for d in eval_term_data]

        if len(all_accuracy_results) == 0:
            accuracy = 0
        else:
            accuracy = sum([1 for d in all_accuracy_results if d]) / len(
                all_accuracy_results
            )

        all_page_results = [d.page_in_range for d in eval_term_data]

        if len(all_page_results) == 0:  
            page_in_range = 0
        else:
            page_in_range = sum([1 for d in all_page_results if d]) / len(all_page_results)

        print("=============================================")
        print(f"Evaluated term: {term}")
        print(f"Accuracy: {accuracy}")
        print("\n")
        print(f"Page Accuracy: {page_in_range}")
        print("=============================================")
        print("\n")


if __name__ == "__main__":
    main()
