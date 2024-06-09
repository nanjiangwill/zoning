from asyncio import run as aiorun

import polars as pl
import typer
from class_types import EvaluationData, EvaluationDataResults, LLMQueries, Place
from hydra import compose, initialize
from llm import VanillaLLM
from omegaconf import OmegaConf
from search import KeywordSearcher
from tqdm.contrib.concurrent import process_map
from typer import Typer


# @hydra.main(version_base=None, config_path="../config", config_name="base")
# hydra decorator was not support by Typer
def main(config_name: str = typer.Argument("base")):
    async def _main():
        with initialize(
            version_base=None, config_path="../config", job_name="test_app"
        ):
            config = compose(config_name=config_name)

        OmegaConf.resolve(config)

        # load searcher
        match config.search.method:
            case "keyword":
                searcher = KeywordSearcher(config)
            case "embedding":
                raise NotImplementedError("Embedding searcher is not implemented yet")
            case _:
                raise ValueError(
                    f"Search method {config.search.method} is not supported"
                )
        match config.llm.method:
            case "vanilla":
                llm = VanillaLLM(config)
            case "few-shot":
                raise NotImplementedError("Few-shot LLM is not implemented yet")
            case _:
                raise ValueError(f"LLM method {config.llm.method} is not supported")

        eval_terms = config.eval_terms

        # The folloing is not the correct way to generate evaluation dataset
        # the reason for doing this is because
        ground_truth = pl.read_csv(
            config.ground_truth_file,
            schema_overrides={
                **{f"{tc}_gt": pl.Utf8 for tc in eval_terms},
                **{f"{tc}_page_gt": pl.Utf8 for tc in eval_terms},
            },
        )
        evaluation_dataset = []
        for row in ground_truth.iter_rows(named=True):
            place = Place(
                town=row["town"],
                district_full_name=row["district"],
                district_short_name=row["district_abb"],
            )
            for eval_term in eval_terms:
                evaluation_data = EvaluationData(
                    place=place,
                    eval_term=eval_term,
                    is_district_fuzzy=searcher.is_district_fuzzy,
                    is_eval_term_fuzzy=searcher.is_eval_term_fuzzy,
                    thesaurus_file=config.thesaurus_file,
                )
                evaluation_dataset.append(evaluation_data)

        # Hack ends here
        # the target is to get evaluation_dataset in correct type

        async def search_and_llm_inference(evaluation_data):
            search_results = searcher.search(evaluation_data.search_pattern)

            llm_inference_results = await llm.query(
                LLMQueries(
                    place=evaluation_data.place,
                    eval_term=evaluation_data.eval_term,
                    search_results=search_results,
                )
            )

            # store the evaluation results
            evaluation_results = EvaluationDataResults(
                place=evaluation_data.place,
                eval_term=evaluation_data.eval_term,
                search_results=search_results,
                llm_inference_results=llm_inference_results,
            )
            evaluation_results.save_to(config.result_output_dir, config.experiment_name)

        # await process_map(search_and_llm_inference, evaluation_dataset)
        # need to change the following to Pool
        await search_and_llm_inference(evaluation_dataset[0])

    aiorun(_main())


if __name__ == "__main__":
    app = Typer(add_completion=False)
    app.command()(main)
    app()
