"""
@Time ： 9/1/2024 1:16 AM
@Auth ： Yizhi Hao
@File ：utils
@IDE ：PyCharm
"""
import json
import glob
from zoning.class_types import (
    EvalResult,
    LLMInferenceResult,
    NormalizedLLMInferenceResult,
    PromptResult,
    SearchResult,
)
from viz_v2.backend.config import Config

def load_evals_data():
    """Load all necessary data at server startup."""
    data_store = {}

    for state, experiment_dir in Config.STATE_EXPERIMENT_MAP.items():
        all_results = {
            k: [
                X.model_construct(**json.loads(open(i).read()))
                for i in sorted(glob.glob(f"{experiment_dir}/{k}/*.json"))
            ]
            for k, X in [
                ("search", SearchResult),
                ("prompt", PromptResult),
                ("llm", LLMInferenceResult),
                ("normalization", NormalizedLLMInferenceResult),
                ("eval", EvalResult),
            ]
        }

        all_eval_terms = sorted(list(set([i.eval_term for i in all_results["eval"]])))
        all_places = sorted(list(set(str(i.place) for i in all_results["eval"])))

        def filtered_by_eval(results, eval_term):
            return {
                k: [result for result in results[k] if result.eval_term == eval_term]
                for k in results
            }

        def filtered_by_place(results, place):
            return {
                k: [result for result in results[k] if str(result.place) == str(place)]
                for k in results
            }

        all_data_by_eval_term = {
            eval_term: {
                place: {"place": place, "eval_term": eval_term}
                | filtered_by_eval(filtered_by_place(all_results, place), eval_term)
                for place in all_places
            }
            for eval_term in all_eval_terms
        }

        data_store[state] = {
            'all_eval_terms': all_eval_terms,
            'all_data_by_eval_term': all_data_by_eval_term
        }

    return data_store
