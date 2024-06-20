import json

import hydra
from omegaconf import OmegaConf

from zoning.class_types import LLMInferenceResults, ZoningConfig
from zoning.eval.evaluator import Evaluator


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """Main function to run the evaluation process based on the provided
    configuration.

     Configs:
        - global_config: GlobalConfig.
        - eval_config: EvalConfig

    Eval Input File Format:
        LLMInferenceResults

    Eval Output File Format:
        EvalMetricByTerm objects for each evaluation term.
    """
    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    eval_config = ZoningConfig(config=config).eval_config

    # Read the input data and read ground truth
    llm_inference_results = LLMInferenceResults.model_construct(
        **json.load(open(global_config.data_flow_llm_file))
    )
    test_data = json.load(open(global_config.test_data_file))

    # Load the Evaluator
    evaluator = Evaluator(eval_config)

    # Run the evaluation
    eval_term_metric_results = [
        evaluator.eval_term_metric(
            eval_term,
            test_data,
            llm_inference_results.llm_inference_results_by_eval_term[eval_term],
        )
        for eval_term in llm_inference_results.llm_inference_results_by_eval_term
    ]

    # Write the output data
    with open(global_config.data_flow_eval_file, "w") as f:
        json.dump(
            [eval_queries.model_dump() for eval_queries, _ in eval_term_metric_results],
            f,
        )
    with open(global_config.data_flow_eval_result_file, "w") as f:
        json.dump(
            [
                eval_metric_by_term.model_dump()
                for _, eval_metric_by_term in eval_term_metric_results
            ],
            f,
        )


if __name__ == "__main__":
    main()
