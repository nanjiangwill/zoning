import copy
import json
import os
from asyncio import run as aiorun
from typing import AsyncGenerator, Tuple
from class_types import Place, LLMQuery, EvaluationData, EvaluationDataResults
import polars as pl
import typer
from hydra import compose, initialize
from llm import VanillaLLM
from omegaconf import OmegaConf
from search import KeywordSearcher
from tqdm.contrib.concurrent import process_map
from typer import Typer

from utils import (
    District,
    EvaluationMetrics,
    flatten,
    page_coverage,
    semantic_comparison,
)

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

        eval_terms = config.eval.terms
        
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
                town=row['town'],
                district_full_name=row['district'],
                district_short_name=row['district_abb']
            )
            for eval_term in config.eval_terms:
                evaluation_data = EvaluationData(
                    place=place,
                    eval_term=eval_term,
                    is_district_fuzzy=searcher.is_district_fuzzy,
                    is_eval_term_fuzzy=searcher.is_eval_term_fuzzy,
                    thesaurus_file=config.thesaurus_file
                )
                evaluation_dataset.append(evaluation_data)
        # Hack ends here
        # the target is to get evaluation_dataset in correct type
        
        
        async def search_and_llm_inference(evaluation_data, searcher, llm):
            search_results = searcher.search(evaluation_data.search_pattern)
            
            # TODO, need a better way to reset the generator
            tmp_search_results = copy.deepcopy(search_results)
            entire_searched_page_range = flatten(page_coverage(tmp_search_results))
            # we just need expanded_pages
            
            llm_inference_results = llm.query(LLMQuery(
                place=place,
                eval_term=eval_term,
                search_results=search_results
            ))
            
            llm_inference_results = []
            async for llm_inference_result in llm_inference_results:
                llm_inference_results.append(llm_inference_result)
            
            return EvaluationDataResults(
                place=place,
                eval_term=eval_term,
                entire_searched_page_range=entire_searched_page_range,
                llm_inference_results=llm_inference_results
            )
            
    aiorun(_main())
    
if __name__ == "__main__":
    app = Typer(add_completion=False)
    app.command()(main)
    app()