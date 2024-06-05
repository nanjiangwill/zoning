from functools import partial
import os
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from .utils import (
    District,
    normalize_town,
    get_town_district_mapping,
    page_coverage,
    flatten,
    AsyncTyper,
    semantic_comparison,
)
from search import KeywordSearcher
from llm import VanillaLLM
import pandas as pd
import polars as pl
from tqdm.contrib.concurrent import process_map
import asyncio
from asyncio import run as aiorun
import copy
import typer
from typer import Typer
import json


async def eval_terms_per_district(district_ground_truth, terms, searcher, llm):
    town = district_ground_truth["town"]
    district = District(
        full_name=district_ground_truth["district"],
        short_name=district_ground_truth["district"],
    )
    for term in terms:
        gt_page = district_ground_truth[f"{term}_page_gt"]
        if gt_page is None:
            # No ground truth page
            gt_page = set()
        else:
            gt_page = set(map(int, str(gt_page).split(",")))

        # true answer
        expected = district_ground_truth[f"{term}_gt"]
        expected_extended = district_ground_truth[f"{term}_gt_orig"]
        is_correct_page_searched = any(gt_page & set(expanded_pages))
        this_correct_page_searched = any(gt_page & set(extracted_pages_expanded))
        is_empty = True

        pages = searcher.search(town, district, term)

        ## TODO, need a better way to reset the generator
        tmp_pages = copy.deepcopy(pages)
        expanded_pages = flatten(page_coverage(tmp_pages))

        llm_output = llm.query(town, district, term, pages)
        async for result in llm_output:
            extracted_pages = {r.page_number for r in result.search_pages}
            extracted_pages_expanded = set(result.search_pages_expanded)
            base_output = {
                "town": town,
                "district": district.full_name,
                "term": term,
                "gt_page": list(gt_page),
                "correct_page_searched": is_correct_page_searched,
                "this_correct_page_searched": this_correct_page_searched,
                "expanded_pages": list(expanded_pages),
                "extracted_pages": list(extracted_pages),
                "extracted_pages_expanded": list(extracted_pages_expanded),
                "expected": expected,
                "expected_extended": expected_extended,
                "pages": [p.page_number for p in pages],
                "confirmed_flag": None,
                "confirmed_raw": None,
                "actual_before_confirmation": None,
            }
            if result.output is None:
                yield {
                    **base_output,
                    "rationale": None,
                    "extracted_text": None,
                    "actual": None,
                }
            else:
                yield {
                    **base_output,
                    "rationale": result.output.rationale,
                    "extracted_text": result.output.extracted_text,
                    "actual": result.output.answer,
                }

        if is_empty:
            yield {
                "town": town,
                "district": district.full_name,
                "term": term,
                "gt_page": list(gt_page),
                "correct_page_searched": False,
                "this_correct_page_searched": False,
                "expanded_pages": None,
                "searched_pages": None,
                "searched_pages_expanded": None,
                "expected": expected,
                "expected_extended": district_ground_truth[f"{term}_gt_orig"],
                "rationale": None,
                "extracted_text": None,
                "actual": None,
                "pages": None,
                "confirmed_flag": None,
                "confirmed_raw": None,
                "actual_before_confirmation": None,
            }


def eval_terms_results(results):
    results_df = pl.from_dicts(results, schema_overrides={"expected_extended": pl.Utf8})
    # 1. page search recall
    search_results_df = results_df.groupby(["town", "district"]).agg(
        [
            pl.col("this_correct_page_searched").sum(),
            pl.col("gt_page").list.lengths().sum(),
        ]
    )

    page_search_correct = len(
        search_results_df.filter(pl.col("this_correct_page_searched") > 0)
    )
    page_search_exists = len(search_results_df.filter(pl.col("gt_page") > 0))
    page_search_recall = page_search_correct / page_search_exists

    # 2. answer accuracy
    answers_df = results_df.with_columns(
        pl.struct(["actual", "expected_extended", "expected"])
        .apply(
            lambda x: semantic_comparison(x["actual"], x["expected_extended"])
            or semantic_comparison(x["actual"], x["expected"])
        )
        .alias("correct_answer")
    )
    answers_results_df = answers_df.groupby(pl.col("town", "district")).agg(
        pl.col("correct_answer").sum(),
    )
    answer_correct = len(answers_results_df.filter(pl.col("correct_answer") > 0))

    # 3. answer + page accuracy
    answers_page_df = answers_df.with_columns(
        pl.struct(["correct_answer", "this_correct_page_searched"])
        .apply(lambda x: x["correct_answer"] and x["this_correct_page_searched"])
        .alias("correct_answer_and_page")
    )
    answers_page_results_df = answers_page_df.groupby(pl.col("town", "district")).agg(
        pl.col("correct_answer_and_page").sum(),
    )
    answer_page_correct = len(
        answers_page_results_df.filter(pl.col("correct_answer_and_page") > 0)
    )

    # 4. answer prec/rec/f1
    # does there exist an answer when the correct page is found?
    pr_answers_df = (
        answers_df.with_columns(
            pl.struct(["this_correct_page_searched", "actual"])
            .apply(lambda x: x["actual"] != "None")
            .alias("predicted_positive")
        )
        .with_columns(
            pl.struct(
                [
                    "this_correct_page_searched",
                    "expected_extended",
                    "expected",
                    "actual",
                ]
            )
            .apply(
                lambda x: x["this_correct_page_searched"]
                and (x["expected_extended"] is not None or x["expected"] is not None)
                and x["actual"] != "None"
            )
            .alias("true_predicted_positive")
        )
        .with_columns(
            pl.struct(["this_correct_page_searched", "actual"])
            .apply(lambda x: x["this_correct_page_searched"] and x["actual"] != "None")
            .alias("positive")
        )
        .with_columns(
            pl.struct(["this_correct_page_searched", "actual"])
            .apply(lambda x: x["this_correct_page_searched"] and x["actual"] == "None")
            .alias("false_negative")
        )
        .with_columns(
            pl.struct(["this_correct_page_searched", "actual"])
            .apply(
                lambda x: not x["this_correct_page_searched"] and x["actual"] != "None"
            )
            .alias("false_positive")
        )
    )
    predicted_positive = pr_answers_df["predicted_positive"].sum()
    true_predicted_positive = pr_answers_df["true_predicted_positive"].sum()
    positive = pr_answers_df["positive"].sum()

    false_positive = pr_answers_df["false_positive"].sum()
    false_negative = pr_answers_df["false_negative"].sum()

    precision = true_predicted_positive / predicted_positive
    recall = true_predicted_positive / positive

    # 5. answer accuracy | correct page
    correct_page_df = answers_df.filter(pl.col("this_correct_page_searched"))
    answer_accuracy_given_correct_page = correct_page_df["correct_answer"].sum() / len(
        correct_page_df
    )

    num_rows = len(search_results_df)
    num_rows_with_answers = page_search_exists

    eval_metrics = {
        "num_results": num_rows,
        "num_row_processed": num_rows,
        "num_row_input": num_rows_with_answers,
        "num_correct_page_searched": page_search_correct,
        "num_correct_answer": answer_correct,
        "row_processed": num_rows,
        "page_search_recall": page_search_recall,
        # This is the answer accuracy conditional on the correct page having
        # been looked up by search
        "conditional_answer_accuracy": answer_accuracy_given_correct_page,
        "answer_accuracy": answer_correct / num_rows,
        "answer_page_accuracy": answer_page_correct / num_rows_with_answers,
        "answer_false_positive": false_positive,
        "answer_false_negative": false_negative,
        "answer_precision": precision,
        "answer_recall": recall,
    }

    # 6. record the wrong answers
    wrong_answers_df = results_df.filter(pl.col("correct_answer") == 0)

    return eval_metrics, pr_answers_df, wrong_answers_df


# @hydra.main(version_base=None, config_path="../config", config_name="base")
# hydra decorator was not support by Typer
def main(config_name: str = typer.Argument("connecticut")):
    async def _main():
        with initialize(
            version_base=None, config_path="../config", job_name="test_app"
        ):
            # config = compose(config_name=config_name)
            config = compose(config_name="connecticut")
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
            case "zero-shot":
                llm = VanillaLLM(config)
            case "few-shot":
                raise NotImplementedError("Few-shot LLM is not implemented yet")
            case _:
                raise ValueError(f"LLM method {config.llm.method} is not supported")

        eval_terms = config.eval.terms
        ground_truth = pl.read_csv(
            config.ground_truth_file,
            schema_overrides={
                **{f"{tc}_gt": pl.Utf8 for tc in eval_terms},
                **{f"{tc}_page_gt": pl.Utf8 for tc in eval_terms},
            },
        )

        results = []
        for row in ground_truth.iter_rows(named=True):
            async for result in eval_terms_per_district(row, eval_terms, searcher, llm):
                results.append(result)

        eval_metrics, pr_answers_df, wrong_answers_df = eval_terms_results(results)

        experiment_name = config.target_state
        metrics_path = os.path.join(
            config.result_output_dir, experiment_name, "metrics.json"
        )
        pr_answers_df_path = os.path.join(
            config.result_output_dir, experiment_name, "pr_answers.csv"
        )
        wrong_answers_df_path = os.path.join(
            config.result_output_dir, experiment_name, "wrong_answers.csv"
        )

        with open(metrics_path, "w") as f:
            json.dump(eval_metrics, f)

        pr_answers_df.to_csv(pr_answers_df_path)
        wrong_answers_df.to_csv(wrong_answers_df_path)

    aiorun(_main())


if __name__ == "__main__":
    app = Typer(add_completion=False)
    app.command()(main)
    app()
    # asyncio.run(main())
