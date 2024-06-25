import re
from typing import List

import hydra
from omegaconf import OmegaConf

from zoning.class_types import (
    LLMInferenceResult,
    NormalizedLLMInferenceResult,
    NormalizedLLMOutput,
    ZoningConfig,
)
from zoning.utils import process


def subtract_numerical_values(answer: str) -> List[str] | None:
    number_pattern = r"\b-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\b"
    numbers_str = re.findall(number_pattern, answer)
    numbers_str = [num.replace(",", "") for num in numbers_str]

    return numbers_str if numbers_str else None


def normalize(data: LLMInferenceResult, target: str) -> NormalizedLLMInferenceResult:
    normalized_llm_outputs = []
    for llm_output in data.llm_outputs:
        normalized_llm_output = NormalizedLLMOutput(
            llm_output=llm_output,
            normalized_answer=(
                subtract_numerical_values(llm_output.answer)
                if llm_output.answer
                else None
            ),
        )
        normalized_llm_outputs.append(normalized_llm_output)
    return NormalizedLLMInferenceResult(
        place=data.place,
        eval_term=data.eval_term,
        search_result=data.search_result,
        input_prompts=data.input_prompts,
        normalized_llm_outputs=normalized_llm_outputs,
    )


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """Main function to run the normalization process based on the provided
    configuration.

    Configs:
        - global_config: GlobalConfig.
        - normalization_config: NormalizationConfig

    Input File Format:
        LLMInferenceResult
        config.llm_dir

    Output File Format:
        NormalizedLLMInferenceResult
        config.normalization_dir
    """
    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    # normalization_config = ZoningConfig(config=config).normalization_config

    num_targets = process(
        global_config.target_eval_file,
        global_config.llm_dir,
        global_config.normalization_dir,
        normalize,
        converter=lambda x: LLMInferenceResult.model_construct(**x),
    )
    print()
    print(f"Stage Normalization Completed. Processed {num_targets} targets. Data saved to {global_config.normalization_dir}.")


if __name__ == "__main__":
    main()
