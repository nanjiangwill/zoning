from jinja2 import Environment, FileSystemLoader
from openai import APIConnectionError, APIError, OpenAI, RateLimitError, Timeout
from tenacity import retry, retry_if_exception_type, wait_random_exponential
from typing import Dict, List
from zoning.class_types import EvalMetricByTerm, EvalQuery, EvalQueries, LLMInferenceResult, EvalConfig
from abc import ABC, abstractmethod

class Evaluator(ABC):
    def __init__(self, eval_config: EvalConfig):
        self.eval_config = eval_config
    
    
    # @cached(cache, lambda *args: json.dumps(args))
    @retry(
        retry=retry_if_exception_type(
            (
                APIError,
                RateLimitError,
                APIConnectionError,
                Timeout,
            )
        ),
        wait=wait_random_exponential(multiplier=1, max=60),
    )
    def semantic_comparison(self, eval_term: str, true_answer: str, predicted: str) -> bool:
        client = OpenAI()
        template = Environment(
            loader=FileSystemLoader(self.eval_config.templates_dir)
        ).get_template("semantic_comparison.pmpt.tpl")
        # TODO: Is there a way to share this implementation with our generic prompt
        # function?
        resp = client.chat.completions.create(
            model="gpt-4-turbo",
            temperature=0.0,  # We want these responses to be deterministic
            max_tokens=1,
            messages=[
                {
                    "role": "user",
                    "content": template.render(
                        predicted=predicted,
                        true_answer=true_answer,
                    ),
                },
            ],
        )
        top_choice = resp.choices[0]
        text = top_choice.message.content
        return text == "Y"
    
    def eval_term_metric(
        self, 
        eval_term: str,
        test_data: List[Dict[str, str]],
        llm_inference_results: List[LLMInferenceResult],
    ) -> List[EvalQueries, EvalMetricByTerm]:
        eval_queries = []
        for d in llm_inference_results:
            d_ground_truth_info = list(
                filter(
                    lambda x: x["town"] == d.place.town
                    and x["district"] == d.place.district_full_name
                    and x["district_abb"] == d.place.district_short_name,
                    test_data,
                )
            )

            # if there is no ground truth for this evaluation data, skip
            if d_ground_truth_info is None:
                continue

            # there show be only one matching for each evaluation data
            d_ground_truth_info = d_ground_truth_info[0]

            eval_query = EvalQuery(
                place=d.place,
                eval_term=eval_term,
                search_result=d.search_result,
                llm_inference_result=d,
                ground_truth=d_ground_truth_info[f"{eval_term}_gt"],
                ground_truth_orig=d_ground_truth_info[f"{eval_term}_gt_orig"],
                ground_truth_page=d_ground_truth_info[f"{eval_term}_page_gt"],
            )
            eval_queries.append(eval_query)

        eval_queries = EvalQueries(eval_queries=eval_queries)

        # TODOL use tool to do evaluation

        # Calculate metrics
        # WIP
        # can add more metrics
        correct_answer_count = total_answer_count =  0
        page_tp = page_fp = 0

        for eval_query in eval_queries.eval_queries:
            answer_correct = False

            for llm_inference_result in eval_query.llm_inference_result.llm_outputs:
                if (
                    llm_inference_result.answer is not None
                    and (
                        self.semantic_comparison(
                            eval_term,
                            llm_inference_result.answer,
                            eval_query.ground_truth,
                        )
                        or self.semantic_comparison(
                            eval_term,
                            llm_inference_result.answer,
                            eval_query.ground_truth_orig,
                        )
                    )
                ):
                    answer_correct = True
                    break

            if answer_correct:
                correct_answer_count += 1
                
            total_answer_count += 1

            # Page evaluation
            if eval_query.ground_truth_page and any([int(i) in eval_query.search_result.entire_search_page_range for i in eval_query.ground_truth_page.split(',')]):
                page_tp += 1
            else:
                page_fp += 1

        eval_metric_by_term = EvalMetricByTerm(
            eval_term=eval_term,
            answer_accuracy=correct_answer_count / total_answer_count,
            page_precision=page_tp / (page_tp + page_fp),
        )

        return eval_queries, eval_metric_by_term
