import glob
import json
import sys

import fitz  # PyMuPDF
import streamlit as st

from zoning.class_types import (
    EvalResult,
    LLMInferenceResult,
    NormalizedLLMInferenceResult,
    Place,
    PromptResult,
    SearchResult,
)
from zoning.utils import flatten, target_pdf

PDF_DIR = f"data/{sys.argv[1]}/pdfs"
EXPERIMENT_DIR = sys.argv[2]  # "results/textract_es_gpt4_connecticut_search_range_3"

st.set_page_config(layout="wide")


# Sidebar config
with st.sidebar:
    # Step 1: upload file

    all_results = {
        k: [
            X.model_construct(**json.load(open(i)))
            for i in sorted(glob.glob(f"{EXPERIMENT_DIR}/{k}/*.json"))
        ]
        for k, X in [
            ("search", SearchResult),
            ("prompt", PromptResult),
            ("llm", LLMInferenceResult),
            ("normalization", NormalizedLLMInferenceResult),
            ("eval", EvalResult),
        ]
    }

    all_eval_terms = list(set([i.eval_term for i in all_results["eval"]]))
    all_places = list(set(str(i.place) for i in all_results["eval"]))

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

    st.write(
        "Number of :orange-background[all] selected data: ",
        len(all_data_by_eval_term),
    )

    # Step 2: Config
    st.divider()
    st.subheader("Step 2: Config", divider="rainbow")

    eval_term = st.radio(
        "Choosing :orange-background[Eval Term] you want to check",
        all_eval_terms,
        key="eval_term",
        index=0,
    )

    eval_type = st.radio(
        "Choosing :orange-background[Data Result Type] you want to check",
        (
            "all",
            "correct",
            "only_wrong_answer",
            "only_wrong_page",
            "wrong_answer_and_page",
        ),
        key="eval_type",
        index=0,
    )

    selected_data = [i for _, i in sorted(all_data_by_eval_term[eval_term].items())]

    if eval_type == "correct":
        selected_data = [
            i
            for i in selected_data
            if len(i["eval"]) != 0
            and any(i["eval"][0].answer_correct)
            and any(i["eval"][0].page_in_range)
        ]
    if eval_type == "only_wrong_answer":
        selected_data = [
            i
            for i in selected_data
            if len(i["eval"]) != 0
            and not any(i["eval"][0].answer_correct)
            and any(i["eval"][0].page_in_range)
        ]
    if eval_type == "only_wrong_page":
        selected_data = [
            i
            for i in selected_data
            if len(i["eval"]) != 0
            and any(i["eval"][0].answer_correct)
            and not any(i["eval"][0].page_in_range)
        ]
    if eval_type == "wrong_answer_and_page":
        selected_data = [
            i
            for i in selected_data
            if len(i["eval"]) == 0
            or (
                not any(i["eval"][0].answer_correct)
                and not any(i["eval"][0].page_in_range)
            )
        ]

    # Step 3: Select one data to check
    st.divider()
    st.subheader("Step 3: Select one data to check", divider="rainbow")
    # st.write(
    #     f"There are total :orange-background[{num_selected_data} selected] evaluation data"
    # )
    print(selected_data)
    place = st.radio(
        "Which evaluation datum to check?", (term["place"] for term in selected_data)
    )

# Load the data for the town.

checked_data = all_data_by_eval_term[eval_term][
    place
]  # list([data for data in selected_data if data["place"] == place])[0]
town, district_short_name, district_full_name = place.split("__")
place = Place(
    town=town,
    district_short_name=district_short_name,
    district_full_name=district_full_name,
)
eval_term = checked_data["eval_term"]
search_result = checked_data["search"][0] if checked_data["search"] else None
prompt_result = checked_data["prompt"][0] if checked_data["prompt"] else None
llm_inference_result = checked_data["llm"][0] if checked_data["llm"] else None
normalized_llm_inference_result = (
    checked_data["normalization"][0] if checked_data["normalization"] else None
)
eval_result = checked_data["eval"][0] if checked_data["eval"] else None

prompt_result = [prompt_result.input_prompts[0]]
raw_model_response = (
    llm_inference_result.llm_outputs[0].raw_model_response
    if llm_inference_result
    else None
)
extracted_text = (
    llm_inference_result.llm_outputs[0].extracted_text if llm_inference_result else None
)
rationale = (
    llm_inference_result.llm_outputs[0].rationale if llm_inference_result else None
)
answer = llm_inference_result.llm_outputs[0].answer if llm_inference_result else None
normalized_llm_inference_result = (
    normalized_llm_inference_result.normalized_llm_outputs[0].normalized_answer
    if normalized_llm_inference_result
    else None
)

entire_search_page_range = (
    search_result.entire_search_page_range if search_result else [0]
)


ground_truth = eval_result.ground_truth if eval_result else None
ground_truth_orig = eval_result.ground_truth_orig if eval_result else None
ground_truth_page = eval_result.ground_truth_page if eval_result else None
answer_correct = eval_result.answer_correct if eval_result else None
page_in_range = eval_result.page_in_range if eval_result else None

pdf_file = target_pdf(place.town, PDF_DIR)
# os.path.join(
#     PDF_DIR,
#     f"{place.town}-zoning-code.pdf",
# )
doc = fitz.open(pdf_file)

jump_pages = entire_search_page_range.copy()

if ground_truth_page:
    if "," in ground_truth_page:
        ground_truth_pages = [int(i) for i in ground_truth_page.split(",")]
        jump_pages.extend(ground_truth_pages)
    else:
        jump_pages.append(ground_truth_page)
jump_pages = [int(i) for i in jump_pages]
jump_pages = sorted(set(jump_pages))  # Remove duplicates and sort


def group_continuous(sorted_list):
    if not sorted_list:
        return []

    result = []
    current_group = [sorted_list[0]]

    for i in range(1, len(sorted_list)):
        if sorted_list[i] == sorted_list[i - 1] + 1:
            current_group.append(sorted_list[i])
        else:
            result.append(current_group)
            current_group = [sorted_list[i]]

    result.append(current_group)
    return result


current_page = min(jump_pages) if jump_pages else 1

# show
st.subheader(f"Town: {place.town}")
st.subheader(f"District: {place.district_full_name}")
st.subheader(f"Eval Term: {eval_term}")
st.divider()


summary_col, search_col = st.columns(2)
with summary_col:
    st.subheader("Core Result Summary")
    st.write(f"Ground Truth: :red-background[{ground_truth}]")
    st.write(f"Ground Truth (Unnormalized): :red-background[{ground_truth_orig}]")
    st.write(f"Ground Truth Page: :red-background[{ground_truth_page}]")
    st.write("\n")
    st.write(f"Is answer correct?: :orange-background[{answer_correct}]")
    st.write("\n")
    st.write(f"Is page in range?: :orange-background[{page_in_range}]")

    # page_slider = st.slider(
    #     "Select page",
    #     min_value=min(jump_pages) if jump_pages else 1,
    #     max_value=max(jump_pages) if jump_pages else len(doc),
    #     value=current_page,
    # )

    current_page = st.number_input(
        "Select page",
        min_value=1,
        max_value=len(doc),
        value=current_page,
    )
    page = doc.load_page(current_page - 1)
    pix = page.get_pixmap()
    img_bytes = pix.pil_tobytes(format="PNG")
    pdf_viewer = st.empty()
    pdf_viewer.image(
        img_bytes,
        caption=f"Page {current_page}",
        use_column_width=True,
    )

with search_col:
    st.subheader("Search & Inference Stats")
    st.write(
        f"entire_search_results_page_range: :orange-background[{sorted(entire_search_page_range)}]"
    )
    st.write(
        f"Normalized LLM answer: :orange-background[{normalized_llm_inference_result}]"
    )
    grouped_jump_pages = group_continuous(jump_pages)
    min_max_grouped_jump_pages = flatten(
        [
            (min(i), max(i)) if min(i) != max(i) else (min(i),)
            for i in grouped_jump_pages
        ]
    )
    # print(flatten(min_max_grouped_jump_pages))
    # if len(min_max_grouped_jump_pages) > 0:
    #     cols = st.columns(2 * len(min_max_grouped_jump_pages) -1)

    # for i in range(len(min_max_grouped_jump_pages)):
    #     page_num = min_max_grouped_jump_pages[i]

    #     cols[2*i].button(
    #         str(page_num),
    #         args=(f"{page_num}",),
    #     )
    #     if i < len(min_max_grouped_jump_pages) - 1:
    #         cols[2*i + 1].write("__")

    st.subheader("LLM Inference Results")

    col7, col8 = st.columns(2)
    with col7:
        st.write("*Answer*")
    with col8:
        st.write("*Full Details*")

    with st.container(height=700, border=False):
        # for idx, llm_inference_result in enumerate(normalized_llm_inference_result):

        #     col7, col8 = st.columns(2)
        #     with col7:
        #         # print(llm_inference_result.keys)
        #         st.write(
        #             "Search Page: :orange-background[{}]".format(
        #                 llm_inference_result.llm_output.search_page_range
        #             )
        #         )
        #         st.write(
        #             "Normalized LLM Answer: :orange-background[{}]".format(
        #                 llm_inference_result.normalized_answer
        #             )
        #         )
        #         st.json(
        #             {
        #                 "extracted_text": llm_inference_result.llm_output.extracted_text,
        #                 "rationale": llm_inference_result.llm_output.rationale,
        #                 "answer": llm_inference_result.llm_output.answer,
        #                 "normalized_answer": llm_inference_result.normalized_answer,
        #             }
        #         )
        #     with col8:
        #         st.json(
        #             {
        #                 "input_prompt": llm_inference_result.llm_output.input_prompt,
        #                 "raw_model_response": llm_inference_result.llm_output.raw_model_response,
        #             },
        #             expanded=False,
        #         )
        #     st.divider()
        col7, col8 = st.columns(2)
        with col7:
            st.write(
                "Search Page: :orange-background[{}]".format(entire_search_page_range)
            )
            st.write(
                "Normalized LLM Answer: :orange-background[{}]".format(
                    normalized_llm_inference_result
                )
            )
            st.json(
                {
                    "extracted_text": extracted_text,
                    "rationale": rationale,
                    "answer": answer,
                    "normalized_answer": normalized_llm_inference_result,
                }
            )
        with col8:
            st.json(
                {
                    "input_prompt": prompt_result,
                    "raw_model_response": raw_model_response,
                },
                expanded=False,
            )
        st.divider()

    st.subheader("Search Results before merge")
    col5, col6 = st.columns(2)
    with col5:
        st.write("*Relevance Score & Page Range*")
    with col6:
        st.write("*Text & Highlight*")
    with st.container(height=400, border=False):
        for idx, search_result in enumerate(search_result.search_matches):
            col5, col6 = st.columns(2)
            with col5:
                st.write(f"Relevance Score :orange-background[{search_result.score}]")
                st.write(
                    f"Page Range :orange-background[{sorted(search_result.page_range)}]"
                )
            with col6:
                st.json(
                    {
                        "text": search_result.text,
                        "highlight": search_result.highlight,
                    },
                    expanded=False,
                )
            st.divider()
