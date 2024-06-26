import json
import os

import fitz  # PyMuPDF
import streamlit as st

from zoning.class_types import DistrictEvalResult

PDF_DIR = "data/connecticut/pdfs"


def jump_page(key):
    st.session_state.current_page = int(key)
    generating_checked_data_view()


def jump_page_from_slider():
    st.session_state.current_page = st.session_state.page_slider
    generating_checked_data_view()


def generating_checked_data_view():
    checked_data = st.session_state.selected_data[
        st.session_state.selected_data_index - 1
    ]
    place = checked_data.place
    eval_term = checked_data.eval_term
    search_result = checked_data.search_result
    entire_search_page_range = search_result.entire_search_page_range
    normalized_llm_outputs = checked_data.normalized_llm_outputs
    ground_truth = checked_data.ground_truth
    ground_truth_orig = checked_data.ground_truth_orig
    ground_truth_page = checked_data.ground_truth_page
    answer_correct = checked_data.answer_correct
    page_in_range = checked_data.page_in_range

    pdf_file = os.path.join(
        PDF_DIR,
        f"{place.town}-zoning-code.pdf",
    )

    st.session_state.doc = fitz.open(pdf_file)

    doc = st.session_state.doc

    jump_pages = entire_search_page_range.copy()
    if ground_truth_page:
        jump_pages.append(ground_truth_page)
    jump_pages = [int(i) for i in jump_pages]
    jump_pages = sorted(set(jump_pages))  # Remove duplicates and sort

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
        st.write(f"Answer Correct: :orange-background[{answer_correct}]")
        st.write(f"Page In Range: :orange-background[{page_in_range}]")

        st.slider(
            "Select page",
            min_value=1,
            max_value=len(doc),
            key="page_slider",
            value=st.session_state.current_page,
            on_change=jump_page_from_slider,
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
        cols = st.columns(len(jump_pages))
        for i, page_num in enumerate(jump_pages):
            cols[i].button(
                str(page_num),
                on_click=jump_page,
                args=(f"{page_num}",),
            )

        st.subheader("Search Results")
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

        st.subheader("LLM Inference Results")

        col7, col8 = st.columns(2)
        with col7:
            st.write("*Answer*")
        with col8:
            st.write("*Full Details*")

        with st.container(height=700, border=False):
            for idx, llm_inference_result in enumerate(normalized_llm_outputs):

                col7, col8 = st.columns(2)
                with col7:
                    # print(llm_inference_result.keys)
                    st.write(
                        "Search Page: :orange-background[{}]".format(
                            llm_inference_result.llm_output.search_page_range
                        )
                    )
                    st.json(
                        {
                            "extracted_text": llm_inference_result.llm_output.extracted_text,
                            "rationale": llm_inference_result.llm_output.rationale,
                            "answer": llm_inference_result.llm_output.answer,
                            "normalized_answer": llm_inference_result.normalized_answer,
                        }
                    )
                with col8:
                    st.json(
                        {
                            "input_prompt": llm_inference_result.llm_output.input_prompt,
                            "raw_model_response": llm_inference_result.llm_output.raw_model_response,
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
        for i in st.session_state.district_eval_results_by_eval_term[
            st.session_state.eval_term
        ]
    ]

    if st.session_state.eval_type == "all":
        st.session_state.selected_data = selected_data
    elif st.session_state.eval_type == "correct":
        st.session_state.selected_data = [
            i for i in selected_data if i.answer_correct and i.page_in_range
        ]
    elif st.session_state.eval_type == "only_wrong_answer":
        st.session_state.selected_data = [
            i for i in selected_data if not i.answer_correct and i.page_in_range
        ]
    elif st.session_state.eval_type == "only_wrong_page":
        st.session_state.selected_data = [
            i for i in selected_data if i.answer_correct and not i.page_in_range
        ]
    elif st.session_state.eval_type == "wrong_answer_and_page":
        st.session_state.selected_data = [
            i for i in selected_data if not i.answer_correct and not i.page_in_range
        ]
    st.session_state.num_selected_data = len(st.session_state.selected_data)


st.set_page_config(layout="wide")

if "doc" not in st.session_state:
    st.session_state.doc = None
if "all_eval_terms" not in st.session_state:
    st.session_state.all_eval_terms = []
if "district_eval_results" not in st.session_state:
    st.session_state.district_eval_results = None
if "num_district_eval_results" not in st.session_state:
    st.session_state.num_district_eval_results = 0
if "district_eval_results_by_eval_term" not in st.session_state:
    st.session_state.district_eval_results_by_eval_term = {}
if "selected_data" not in st.session_state:
    st.session_state.selected_data = []
if "num_selected_data" not in st.session_state:
    st.session_state.num_selected_data = 0
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# Sidebar config
with st.sidebar:
    # Step 1: upload file
    st.subheader(
        "Step 1: Select files in eval folder",
        divider="rainbow",
    )
    uploaded_files = st.file_uploader(
        "You can find files needed under *data/<state>/eval*",
        type="json",
        on_change=reset,
        accept_multiple_files=True,
    )
    if uploaded_files is not None:
        district_eval_results = [
            DistrictEvalResult.model_construct(**json.load(uploaded_file))
            for uploaded_file in uploaded_files
        ]

        st.session_state.district_eval_results = district_eval_results
        st.session_state.num_district_eval_results = len(district_eval_results)
        st.write(
            "Number of :orange-background[all] selected data: ",
            st.session_state.num_district_eval_results,
        )

    # Step 2: Config
    st.divider()
    st.subheader("Step 2: Config", divider="rainbow")
    st.session_state.all_eval_terms = list(
        set([i.eval_term for i in district_eval_results])
    )
    st.session_state.district_eval_results_by_eval_term = {
        eval_term: [i for i in district_eval_results if i.eval_term == eval_term]
        for eval_term in st.session_state.all_eval_terms
    }

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
        index=None,
        on_change=get_selected_data,
    )

    # Step 3: Select one data to check
    st.divider()
    st.subheader("Step 3: Select one data to check", divider="rainbow")
    st.write(
        f"There are total :orange-background[{st.session_state.num_selected_data} selected] evaluation data"
    )
    st.number_input(
        "Which evaluation datum to check?",
        key="selected_data_index",
        min_value=0,
        max_value=st.session_state.num_selected_data,
        value=None,
        step=1,
        on_change=generating_checked_data_view,
    )
