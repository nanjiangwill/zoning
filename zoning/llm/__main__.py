import json
from asyncio import run as aiorun

import typer
from hydra import compose, initialize
from omegaconf import OmegaConf
from typer import Typer

from zoning.class_types import SearchResults, ZoningConfig
from zoning.llm.vanilla_llm import VanillaLLM


def main(config_name: str = typer.Argument("base")):
    """Main function to run the llm inference based on the provided
    configuration.

    Configs:
        - global_config: GlobalConfig.
        - llm_config: LLMConfig

    LLM Input File Format:
        SearchResults

    LLM Output File Format:
        LLMInferenceResults
    """

    async def _main():
        # Parse the config
        # async function does not work well with hydra, so we use the initialize function
        with initialize(
            version_base=None, config_path="../../config", job_name="test_app"
        ):
            config = compose(config_name=config_name)

        config = OmegaConf.to_object(config)
        global_config = ZoningConfig(config=config).global_config
        llm_config = ZoningConfig(config=config).llm_config

        # Read the input data
        with open(global_config.data_flow_search_file, "r") as f:
            data_string = json.load(f)
            search_results = SearchResults.model_construct(**json.loads(data_string))

        # Load the searcher
        match llm_config.method:
            case "vanilla":
                llm = VanillaLLM(llm_config)
            case "few-shot":
                raise NotImplementedError("Few-shot LLM is not implemented yet")
            case _:
                raise ValueError(f"LLM method {llm_config.method} is not supported")

        llm_inference_results = await llm.query(search_results)

        # Write the output data, data type is SearchResults
        with open(global_config.data_flow_llm_file, "w") as f:
            json.dump(llm_inference_results.model_dump_json(), f)

    aiorun(_main())


if __name__ == "__main__":
    app = Typer(add_completion=False)
    app.command()(main)
    app()
