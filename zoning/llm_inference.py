import asyncio
import random
from asyncio import run as aiorun

import polars as pl
import tqdm
import typer
from class_types import (
    AllEvaluationResults,
    EvaluationDatum,
    EvaluationDatumResult,
    LLMQueries,
    Place,
)
from hydra import compose, initialize
from llm import LLM, VanillaLLM
from omegaconf import OmegaConf
from search import KeywordSearcher, Searcher
from typer import Typer


async def search_and_llm_inference(
    evaluation_datum: EvaluationDatum, searcher: Searcher, llm: LLM
) -> EvaluationDatumResult | None:
    try:
        search_results = searcher.search(evaluation_datum.search_pattern)
        llm_inference_results = await llm.query(
            LLMQueries(
                place=evaluation_datum.place,
                eval_term=evaluation_datum.eval_term,
                search_results=search_results,
            )
        )

        # store the evaluation results
        return EvaluationDatumResult(
            place=evaluation_datum.place,
            eval_term=evaluation_datum.eval_term,
            search_results=search_results,
            llm_inference_results=llm_inference_results,
        )
    except Exception as e:
        print(f"Error processing {evaluation_datum.place} {evaluation_datum.eval_term}")
        print(e)
        return None


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
                evaluation_datum = EvaluationDatum(
                    place=place,
                    eval_term=eval_term,
                    is_district_fuzzy=searcher.is_district_fuzzy,
                    is_eval_term_fuzzy=searcher.is_eval_term_fuzzy,
                    thesaurus_file=searcher.thesaurus_file,
                )
                evaluation_dataset.append(evaluation_datum)

        # Hack ends here
        # the target is to get evaluation_dataset in correct type

        if config.random_seed and (config.test_size or config.test_percentage):
            random.seed(config.random_seed)
            random.shuffle(evaluation_dataset)
            evaluation_dataset = evaluation_dataset[: config.test_size]
        else:
            raise ValueError("random_seed and test_size must be provided to save cost")

        async_tasks = [
            search_and_llm_inference(evaluation_data, searcher=searcher, llm=llm)
            for evaluation_data in evaluation_dataset
        ]

        async_results = []

        pbar = tqdm.tqdm(total=len(async_tasks))
        for f in asyncio.as_completed(async_tasks):
            async_result = await f
            pbar.update()
            if async_result is not None:
                pbar.set_description(
                    f"Processing {async_result.place} {async_result.eval_term}"
                )
                async_results.append(async_result)
        pbar.close()
        # potential implementation of multiprocessing
        # loop = asyncio.get_running_loop()

        # with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        #     tasks = [
        #         loop.run_in_executor(executor, search_and_llm_inference_partial, evaluation_datum)
        #         for evaluation_datum in evaluation_dataset
        #     ]
        #     multiprocess_results = []
        #     pbar = tqdm(total=len(tasks))
        #     for f in asyncio.as_completed(tasks):
        #         value = await f
        #         # pbar.set_description(value)
        #         multiprocess_results.append(value)
        #         pbar.update()
        #     pbar.close()
        all_evaluation_results = AllEvaluationResults(
            all_evaluation_results=async_results
        )
        all_evaluation_results.save_to(config.result_output_dir)

    aiorun(_main())


if __name__ == "__main__":
    app = Typer(add_completion=False)
    app.command()(main)
    app()
