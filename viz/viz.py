import glob
import json
import os

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
from zoning.utils import flatten

PDF_DIR = "data/connecticut/pdfs"
EXPERIMENT_DIR = "results/textract_es_gpt4_connecticut_search_range_3"


def jump_page(key):
    st.session_state.current_page = int(key)
    generating_checked_data_view()


def jump_page_from_slider():
    st.session_state.current_page = st.session_state.page_slider
    generating_checked_data_view()


def jump_page_from_number_input():
    st.session_state.current_page = st.session_state.page_number_input
    generating_checked_data_view()


def generating_checked_data_view():
    if len(st.session_state.selected_data) == 0:
        return
    checked_data = st.session_state.selected_data[
        st.session_state.selected_data_index - 1
    ]
    town, district_short_name, district_full_name = checked_data["place"].split("__")
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
        llm_inference_result.llm_outputs[0].extracted_text
        if llm_inference_result
        else None
    )
    rationale = (
        llm_inference_result.llm_outputs[0].rationale if llm_inference_result else None
    )
    answer = (
        llm_inference_result.llm_outputs[0].answer if llm_inference_result else None
    )
    normalized_llm_inference_result = (
        normalized_llm_inference_result.normalized_llm_outputs[0].normalized_answer
        if normalized_llm_inference_result
        else None
    )

    entire_search_page_range = (
        search_result.entire_search_page_range if search_result else []
    )
    # if search_result:
    #     normalized_llm_inference_result = sorted(
    #         normalized_llm_inference_result,
    #         key=lambda x: len(x.llm_output.search_match),
    #         reverse=True,
    #     )
    ground_truth = eval_result.ground_truth if eval_result else None
    ground_truth_orig = eval_result.ground_truth_orig if eval_result else None
    ground_truth_page = eval_result.ground_truth_page if eval_result else None
    answer_correct = eval_result.answer_correct if eval_result else None
    page_in_range = eval_result.page_in_range if eval_result else None

    pdf_file = os.path.join(
        PDF_DIR,
        f"{place.town}-zoning-code.pdf",
    )
    st.session_state.doc = fitz.open(pdf_file)

    doc = st.session_state.doc

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

    if st.session_state.current_page == 0:
        st.session_state.current_page = min(jump_pages)
    # show
    st.subheader(f"Place: :orange-background[{place.town}-{place.district_full_name}]")
    st.subheader(f"eval_term: :orange-background[{eval_term}]")
    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Ground Truth Stats")
        st.write(f"Ground Truth Answer: :orange-background[{ground_truth}]")
        st.write(f"Ground Truth Orig: :orange-background[{ground_truth_orig}]")
        st.write(f"Ground Truth Page: :orange-background[{ground_truth_page}]")
        st.write("\n")
        st.write(f"Answer Correct: :orange-background[{answer_correct}]")
        st.write("\n")
        st.write(f"Page In Range: :orange-background[{page_in_range}]")

        st.slider(
            "Select page",
            min_value=min(jump_pages) if jump_pages else 1,
            max_value=max(jump_pages) if jump_pages else len(doc),
            key="page_slider",
            value=st.session_state.current_page,
            on_change=jump_page_from_slider,
        )
        st.number_input(
            "Select page",
            min_value=1,
            max_value=len(doc),
            key="page_number_input",
            value=st.session_state.current_page,
            on_change=jump_page_from_number_input,
        )
        page = doc.load_page(st.session_state.current_page - 1)
        pix = page.get_pixmap()
        img_bytes = pix.pil_tobytes(format="PNG")
        if "pdf_viewer" not in st.session_state:
            st.session_state.pdf_viewer = st.empty()
        st.session_state.pdf_viewer.image(
            img_bytes,
            caption=f"Page {st.session_state.current_page}",
            use_column_width=True,
        )

    with col4:
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
                (min(i), max(i)) if min(i) != max(i) else min(i)
                for i in grouped_jump_pages
            ]
        )
        # print(flatten(min_max_grouped_jump_pages))
        if len(min_max_grouped_jump_pages) > 0:
            cols = st.columns(2 * len(min_max_grouped_jump_pages) -1)
            
        for i in range(len(min_max_grouped_jump_pages)):
            page_num = min_max_grouped_jump_pages[i]
            
            cols[2*i].button(
                str(page_num),
                on_click=jump_page,
                args=(f"{page_num}",),
            )
            if i < len(min_max_grouped_jump_pages) - 1:
                cols[2*i + 1].write("__")

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
                    "Search Page: :orange-background[{}]".format(
                        entire_search_page_range
                    )
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
                    st.write(
                        f"Relevance Score :orange-background[{search_result.score}]"
                    )
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


def reset():
    for key in st.session_state.keys():
        del st.session_state[key]


def get_selected_data():
    selected_data = [
        i
        for i in st.session_state.all_data_by_eval_term[
            st.session_state.eval_term
        ].values()
    ]

    if st.session_state.eval_type == "all":
        st.session_state.selected_data = selected_data
    elif st.session_state.eval_type == "correct":
        st.session_state.selected_data = [
            i
            for i in selected_data
            if len(i["eval"]) != 0
            and any(i["eval"][0].answer_correct)
            and any(i["eval"][0].page_in_range)
        ]
    elif st.session_state.eval_type == "only_wrong_answer":
        st.session_state.selected_data = [
            i
            for i in selected_data
            if len(i["eval"]) != 0
            and not any(i["eval"][0].answer_correct)
            and any(i["eval"][0].page_in_range)
        ]
    elif st.session_state.eval_type == "only_wrong_page":
        st.session_state.selected_data = [
            i
            for i in selected_data
            if len(i["eval"]) != 0
            and any(i["eval"][0].answer_correct)
            and not any(i["eval"][0].page_in_range)
        ]
    elif st.session_state.eval_type == "wrong_answer_and_page":
        st.session_state.selected_data = [
            i
            for i in selected_data
            if len(i["eval"]) == 0
            or (
                not any(i["eval"][0].answer_correct)
                and not any(i["eval"][0].page_in_range)
            )
        ]
    st.session_state.num_selected_data = len(st.session_state.selected_data)
    st.session_state.current_page = 0


st.set_page_config(layout="wide")

if "doc" not in st.session_state:
    st.session_state.doc = None
if "all_eval_terms" not in st.session_state:
    st.session_state.all_eval_terms = []
if "all_data_by_eval_terms" not in st.session_state:
    st.session_state.all_data_by_eval_terms = {}
if "num_all_data" not in st.session_state:
    st.session_state.num_all_data = 0
if "selected_data" not in st.session_state:
    st.session_state.selected_data = []
if "num_selected_data" not in st.session_state:
    st.session_state.num_selected_data = 0
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
if "init_show" not in st.session_state:
    st.session_state.init_show = True

# Sidebar config
with st.sidebar:
    # Step 1: upload file
    st.subheader(
        "Step 1: Select files in eval folder",
        divider="rainbow",
    )
    # uploaded_files = st.file_uploader(
    #     "You can find files needed under *results/<experiment_name>/eval*",
    #     type="json",
    #     on_change=reset,
    #     accept_multiple_files=True,
    # )

    all_search_results = [
        SearchResult.model_construct(**json.load(open(i)))
        for i in glob.glob(f"{EXPERIMENT_DIR}/search/*.json")
    ]
    all_prompt_results = [
        PromptResult.model_construct(**json.load(open(i)))
        for i in glob.glob(f"{EXPERIMENT_DIR}/prompt/*.json")
    ]
    all_llm_results = [
        LLMInferenceResult.model_construct(**json.load(open(i)))
        for i in glob.glob(f"{EXPERIMENT_DIR}/llm/*.json")
    ]
    all_normalized_llm_results = [
        NormalizedLLMInferenceResult.model_construct(**json.load(open(i)))
        for i in glob.glob(f"{EXPERIMENT_DIR}/normalization/*.json")
    ]
    all_eval_results = [
        EvalResult.model_construct(**json.load(open(i)))
        for i in glob.glob(f"{EXPERIMENT_DIR}/eval/*.json")
    ]

    all_eval_terms = list(set([i.eval_term for i in all_eval_results]))
    all_places = list(set(str(i.place) for i in all_eval_results))

    all_data_by_place = {
        place: {
            eval_term: {
                "search": [
                    i
                    for i in all_search_results
                    if i.eval_term == eval_term and str(i.place) == str(place)
                ],
                "prompt": [
                    i
                    for i in all_prompt_results
                    if i.eval_term == eval_term and str(i.place) == str(place)
                ],
                "llm": [
                    i
                    for i in all_llm_results
                    if i.eval_term == eval_term and str(i.place) == str(place)
                ],
                "normalization": [
                    i
                    for i in all_normalized_llm_results
                    if i.eval_term == eval_term and str(i.place) == str(place)
                ],
                "eval": [
                    i
                    for i in all_eval_results
                    if i.eval_term == eval_term and str(i.place) == str(place)
                ],
            }
            for eval_term in all_eval_terms
        }
        for place in all_places
    }

    all_data_by_eval_term = {
        eval_term: {
            place: {
                "place": place,
                "eval_term": eval_term,
                "search": [
                    i
                    for i in all_search_results
                    if i.eval_term == eval_term and str(i.place) == str(place)
                ],
                "prompt": [
                    i
                    for i in all_prompt_results
                    if i.eval_term == eval_term and str(i.place) == str(place)
                ],
                "llm": [
                    i
                    for i in all_llm_results
                    if i.eval_term == eval_term and str(i.place) == str(place)
                ],
                "normalization": [
                    i
                    for i in all_normalized_llm_results
                    if i.eval_term == eval_term and str(i.place) == str(place)
                ],
                "eval": [
                    i
                    for i in all_eval_results
                    if i.eval_term == eval_term and str(i.place) == str(place)
                ],
            }
            for place in all_places
        }
        for eval_term in all_eval_terms
    }

    st.session_state.all_data_by_eval_term = all_data_by_eval_term
    st.session_state.num_all_data = len(all_eval_results)
    st.write(
        "Number of :orange-background[all] selected data: ",
        st.session_state.num_all_data,
    )

    # Step 2: Config
    st.divider()
    st.subheader("Step 2: Config", divider="rainbow")
    st.session_state.all_eval_terms = all_eval_terms

    st.radio(
        "Choosing :orange-background[Eval Term] you want to check",
        st.session_state.all_eval_terms,
        key="eval_term",
        index=0,
        on_change=get_selected_data,
    )

    st.radio(
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
        on_change=get_selected_data,
    )
    get_selected_data()

    # Step 3: Select one data to check
    st.divider()
    st.subheader("Step 3: Select one data to check", divider="rainbow")
    st.write(
        f"There are total :orange-background[{st.session_state.num_selected_data} selected] evaluation data"
    )
    st.number_input(
        "Which evaluation datum to check?",
        key="selected_data_index",
        min_value=1,
        max_value=st.session_state.num_selected_data,
        value=1,
        step=1,
        on_change=generating_checked_data_view,
    )
if st.session_state.init_show:
    generating_checked_data_view()
    st.session_state.init_show = False
