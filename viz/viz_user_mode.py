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
from zoning.utils import target_pdf

PDF_DIR = f"data/{sys.argv[1]}/pdfs"
EXPERIMENT_DIR = sys.argv[2]  # "results/textract_es_gpt4_connecticut_search_range_3"

st.set_page_config(layout="wide")


def write_data(human_feedback: str):
    try:
        existing_data = json.load(open("data.json"))
    except FileNotFoundError or json.decoder.JSONDecodeError:
        existing_data = []

    d = {
        "place": str(place),
        "eval_term": eval_term,
        "human_feedback": human_feedback,
    }
    existing_data.append(d)
    with open("data.json", "w") as f:
        json.dump(existing_data, f)


# Sidebar config
with st.sidebar:
    # Step 1: load files

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
        len(all_places),
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
            if i["eval"][0].answer_correct and i["eval"][0].page_in_range
        ]
    if eval_type == "only_wrong_answer":
        selected_data = [
            i
            for i in selected_data
            if not i["eval"][0].answer_correct and i["eval"][0].page_in_range
        ]
    if eval_type == "only_wrong_page":
        selected_data = [
            i
            for i in selected_data
            if i["eval"][0].answer_correct and not i["eval"][0].page_in_range
        ]
    if eval_type == "wrong_answer_and_page":
        selected_data = [
            i
            for i in selected_data
            if not i["eval"][0].answer_correct and not i["eval"][0].page_in_range
        ]

    # Step 3: Select one data to check
    st.divider()
    st.subheader("Step 3: Select one data to check", divider="rainbow")
    place = st.radio("Which data to check?", (term["place"] for term in selected_data))

# Load the data for the town.

visualized_data = all_data_by_eval_term[eval_term][
    place
]  # list([data for data in selected_data if data["place"] == place])[0]
place = Place.from_str(place)

# loading info
eval_term = visualized_data["eval_term"]
search_result = visualized_data["search"][0]
prompt_result = visualized_data["prompt"][0]
input_prompt = prompt_result.input_prompts[0]
llm_inference_result = visualized_data["llm"][0]
normalized_llm_inference_result = visualized_data["normalization"][0]
eval_result = visualized_data["eval"][0]

llm_output = llm_inference_result.llm_outputs[0]
normalized_llm_output = normalized_llm_inference_result.normalized_llm_outputs[0]


entire_search_page_range = search_result.entire_search_page_range

highlight_text_pages = []
if llm_output.extracted_text is not None:
    for i in llm_output.extracted_text:
        # x = repr(i)
        x = i
        page = int(
            input_prompt.user_prompt.split(x)[0].split("NEW PAGE ")[-1].split("\n")[0]
        )
        highlight_text_pages.append(page)

    highlight_text_pages = sorted(list(set(highlight_text_pages)))

ground_truth = eval_result.ground_truth
ground_truth_orig = eval_result.ground_truth_orig
ground_truth_page = eval_result.ground_truth_page
answer_correct = eval_result.answer_correct
page_in_range = eval_result.page_in_range

pdf_file = target_pdf(place.town, PDF_DIR)
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
st.subheader(f"District Abbreviation: {place.district_short_name}")
st.subheader(f"Eval Term: {eval_term}")
st.divider()


summary_col, search_col = st.columns(2)
with summary_col:

    st.subheader(
        "Select the page to view by clicking the page button or giving a page number"
    )

    cols = st.columns(len(jump_pages))
    for i in range(len(jump_pages)):
        page_num = jump_pages[i]
        if cols[i].button(
            str(page_num),
            args=(f"{page_num}",),
        ):
            current_page = page_num
            # st.rerun()
    current_page = st.number_input(
        "",
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
    st.title("Results")
    st.subheader("Search & Inference Stats")
    st.write(f"Search Pages: :orange-background[{sorted(entire_search_page_range)}]")

    st.write(
        f"LLM Answer: :orange-background[{normalized_llm_output.normalized_answer}]"
    )
    st.write(f"LLM Answer appears in page: :orange-background[{highlight_text_pages}]")
    st.write(f"LLM Extracted Text: :orange-background[{llm_output.extracted_text}]")
    st.write(f"LLM Rationale: :orange-background[{llm_output.rationale}]")

    correct_col, not_sure_col, wrong_col = st.columns(3)
    with correct_col:
        if st.button(
            "LLM Correct",
            key="llm_correct",
            type="secondary",
        ):
            write_data("correct")
    with not_sure_col:
        if st.button(
            "Not Sure",
            key="llm_not_sure",
            type="secondary",
        ):
            write_data("not_sure")
    with wrong_col:
        if st.button(
            "LLM Wrong",
            key="llm_wrong",
            type="secondary",
        ):
            write_data("wrong")

    st.divider()

    st.title("More Details")
    with st.container(height=700):
        st.subheader("LLM Inference Results")

        llm_answer_col, llm_response_detail_col = st.columns(2)
        with llm_answer_col:
            st.write("*Answer*")
        with llm_response_detail_col:
            st.write("*Full Details*")

        with st.container(height=700, border=False):
            llm_answer_col, llm_response_detail_col = st.columns(2)
            with llm_answer_col:
                st.write(
                    "Search Page: :orange-background[{}]".format(
                        entire_search_page_range
                    )
                )
                st.write(
                    "Highlighted Pages: :orange-background[{}]".format(
                        highlight_text_pages
                    )
                )
                st.write(
                    "Normalized LLM Answer: :orange-background[{}]".format(
                        normalized_llm_output.normalized_answer
                    )
                )
                st.json(
                    {
                        "extracted_text": llm_output.extracted_text,
                        "rationale": llm_output.rationale,
                        "answer": llm_output.answer,
                        "normalized_answer": normalized_llm_output.normalized_answer,
                    }
                )
            with llm_response_detail_col:
                st.json(
                    {
                        "input_prompt": [
                            input_prompt.system_prompt,
                            input_prompt.user_prompt,
                        ],
                        "raw_model_response": llm_output.raw_model_response,
                    },
                    expanded=False,
                )
            st.divider()

        st.subheader("Search Results before merge")
        search_meta_col, search_text_col = st.columns(2)
        with search_meta_col:
            st.write("*Relevance Score & Page Range*")
        with search_text_col:
            st.write("*Text & Highlight*")
        with st.container(height=400, border=False):
            for idx, search_result in enumerate(search_result.search_matches):
                search_meta_col, search_text_col = st.columns(2)
                with search_meta_col:
                    st.write(
                        f"Relevance Score :orange-background[{search_result.score}]"
                    )
                    st.write(
                        f"Page Range :orange-background[{sorted(search_result.page_range)}]"
                    )
                with search_text_col:
                    st.json(
                        {
                            "text": search_result.text,
                            "highlight": search_result.highlight,
                        },
                        expanded=False,
                    )
                st.divider()
