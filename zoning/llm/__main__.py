from asyncio import run as aiorun

import typer
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from typer import Typer

from zoning.class_types import PromptResult, ZoningConfig
from zoning.llm.base_llm import LLM
from zoning.utils import process_async


def main(config_name: str = typer.Argument("base")):
    """Main function to run the llm based on the provided configuration.

    Configs:
        - global_config: GlobalConfig.
        - llm_config: LLMConfig

    Input File Format:
        PromptResult
        config.prompt_dir

    Output File Format:
        LLMInferenceResult
        config.llm_dir
    """

    async def _main():
        # Parse the config
        # async function does not work well with hydra, so we use the initialize function
        GlobalHydra.instance().clear()
        with initialize(
            version_base=None, config_path="../../config", job_name="test_app"
        ):
            config = compose(config_name=config_name)

        config = OmegaConf.to_object(config)
        global_config = ZoningConfig(config=config).global_config
        llm_config = ZoningConfig(config=config).llm_config

        # Load the searcher
        llm = LLM(llm_config)

        await process_async(
            global_config.target_eval_file,
            global_config.prompt_dir,
            global_config.llm_dir,
            llm.query,
            converter=lambda x: PromptResult.model_construct(**x),
        )

    aiorun(_main())


if __name__ == "__main__":
    app = Typer(add_completion=False)
    app.command()(main)
    app()
