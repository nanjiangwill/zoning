# tmp code for score

        # ground_truth = pl.read_csv(
        #     config.ground_truth_file,
        #     schema_overrides={
        #         **{f"{tc}_gt": pl.Utf8 for tc in eval_terms},
        #         **{f"{tc}_page_gt": pl.Utf8 for tc in eval_terms},
        #     },
        # )

        # results = []

        # # Change the following to use process_map
        # for row in ground_truth.iter_rows(named=True):
        #     async for result in eval_terms_per_district(row, eval_terms, searcher, llm):
        #         results.append(result)

        # eval_metrics, pr_answers_df, wrong_answers_df = eval_terms_results(results)

        # experiment_name = config.target_state
        # metrics_path = os.path.join(
        #     config.result_output_dir, experiment_name, "metrics.json"
        # )
        # pr_answers_df_path = os.path.join(
        #     config.result_output_dir, experiment_name, "pr_answers.csv"
        # )
        # wrong_answers_df_path = os.path.join(
        #     config.result_output_dir, experiment_name, "wrong_answers.csv"
        # )

        # with open(metrics_path, "w") as f:
        #     json.dump(eval_metrics, f)

        # pr_answers_df.to_csv(pr_answers_df_path)
        # wrong_answers_df.to_csv(wrong_answers_df_path)