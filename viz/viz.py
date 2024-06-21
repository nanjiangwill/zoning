import base64
import json
import os

import streamlit as st

from zoning.class_types import DistrictEvalResult

PDF_DIR = "data/connecticut/pdfs"


# @st.cache_data
def load_pdf(pdf_file: str):
    with open(pdf_file, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# @st.cache_data
def go_to_pdf_page():
    print("inside go_to_pdf_page")
    st.session_state.selected_pdf_page = 40
    st.session_state.display_pdf = f"""
    <iframe src="data:application/pdf;base64,{st.session_state.base64_pdf}#page=40" width="550" height="900" type="application/pdf"></iframe>
"""


def generating_checked_data_view():
    checked_data = st.session_state.selected_data[
        st.session_state.selected_data_index - 1
    ]
    place = checked_data.place
    eval_term = checked_data.eval_term
    search_result = checked_data.search_result
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
    if "selected_pdf_page" not in st.session_state:
        st.session_state.selected_pdf_page = 0
    if "base64_pdf" not in st.session_state:
        st.session_state.base64_pdf = load_pdf(pdf_file)

    if "display_pdf" not in st.session_state:
        st.session_state.display_pdf = f"""
    <iframe src="data:application/pdf;base64,{st.session_state.base64_pdf}" width="550" height="900" type="application/pdf"></iframe>
"""
    elif st.session_state.selected_pdf_page != 0:
        st.session_state.display_pdf = f"""
    <iframe src="data:application/pdf;base64,{st.session_state.base64_pdf}#page={st.session_state.selected_pdf_page}" width="550" height="900" type="application/pdf"></iframe>
"""

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
        st.markdown(st.session_state.display_pdf, unsafe_allow_html=True)
    with col4:
        st.subheader("Search & Inference Stats")
        st.write(
            f"entire_search_results_page_range: :orange-background[{sorted(search_result.entire_search_results_page_range)}]"
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
                            llm_inference_result.search_page_range
                        )
                    )
                    st.json(
                        {
                            "extracted_text": llm_inference_result.extracted_text,
                            "rationale": llm_inference_result.rationale,
                            "answer": llm_inference_result.answer,
                        }
                    )
                with col8:
                    st.json(
                        {
                            "input_prompt": llm_inference_result.input_prompt,
                            "raw_model_response": llm_inference_result.raw_model_response,
                        },
                        expanded=False,
                    )
                st.divider()


def reset():
    for key in st.session_state.keys():
        del st.session_state[key]


def process_evaluation_results():
    assert (
        st.session_state.all_evaluation_results
        and st.session_state.num_all_evaluation_data > 0
    )
    assert st.session_state.evaluation_type in [
        "all",
        "correct",
        "only_wrong_answer",
        "only_wrong_page",
        "wrong_answer_and_page",
    ], "Invalid evaluation type"

    st.session_state.current_evaluation_results = (
        st.session_state.all_evaluation_results
    )
    if st.session_state.not_including_gt_is_none:

        st.session_state.current_evaluation_results = [
            i
            for i in st.session_state.all_evaluation_results
            if i.ground_truth is not None
        ]

    if st.session_state.evaluation_type == "all":
        st.session_state.num_current_evaluation_data = len(
            st.session_state.current_evaluation_results
        )
    else:
        raise NotImplementedError("Not implemented yet")


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


def main():
    st.set_page_config(layout="wide")

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

    # Sidebar config
    with st.sidebar:
        # Step 1: upload file
        st.subheader(
            "Step 1: Select files in eval folder",
            divider="rainbow",
        )
        uploaded_files = st.file_uploader(
            "You can find this file under *data/<state>/eval*",
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
            index=0,
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

        # if uploaded_file is not None:
        #     all_evaluation_results = json.load(uploaded_file)
        #     print(all_evaluation_results)

        #     all_evaluation_results = [
        #         EvaluationDatumResult(**json.loads(i)) for i in all_evaluation_results
        #     ]
        #     st.session_state.all_evaluation_results = all_evaluation_results
        #     st.session_state.num_all_evaluation_data = len(all_evaluation_results)
        #     st.write(
        #         "Number of :orange-background[all] evaluation data: ",
        #         st.session_state.num_all_evaluation_data,
        #     )

        # # Step 2: Config
        # st.divider()
        # st.subheader("Step 2: Config", divider="rainbow")
        # st.radio(
        #     "Choosing :orange-background[Evaluation Datum Category] you want to check",
        #     (
        #         "all",
        #         "correct",
        #         "only_wrong_answer",
        #         "only_wrong_page",
        #         "wrong_answer_and_page",
        #     ),
        #     key="evaluation_type",
        #     index=0,
        # )
        # st.toggle(
        #     "Not including ground truth which is None", key="not_including_gt_is_none"
        # )

        # st.button("Apply Changes", on_click=process_evaluation_results)

        # # Step 3: Select one evaluation datum to check
        # st.divider()
        # st.subheader("Step 3: Select one evaluation datum to check", divider="rainbow")
        # st.write(
        #     f"There are total :orange-background[{st.session_state.num_current_evaluation_data} selected] evaluation data"
        # )
        # st.number_input(
        #     "Which evaluation datum to check?",
        #     key="evaluation_datum_index",
        #     min_value=0,
        #     max_value=st.session_state.num_current_evaluation_data,
        #     value=None,
        #     step=1,
        #     on_change=generating_evaluation_datum_view,
        # )


if __name__ == "__main__":
    main()


# import base64

# import streamlit as st
# from streamlit.components.v1 import html


# def load_pdf(pdf_file: str):
#     with open(pdf_file, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")


# if "base64_pdf" not in st.session_state:
#     st.session_state.base64_pdf = load_pdf(
#         "data/connecticut/pdfs/east-hartford-zoning-code.pdf"
#     )

# if "page" not in st.session_state:
#     st.session_state.page = 0

# if "display_pdf" not in st.session_state:
#     st.session_state.display_pdf = f"""
#     <embed src="data:application/pdf;base64,{st.session_state.base64_pdf}#page={st.session_state.page}" width="550" height="900" type="application/pdf"></iframe>
# """

# # placeholder = st.empty()

# # with placeholder.container():
# html(st.session_state.display_pdf, height=900)

# # if st.button("Go to page **5**"):
# #     st.session_state.page = 5
# #     st.session_state.display_pdf = f"""
# #     <embed src="data:application/pdf;base64,{st.session_state.base64_pdf}#page={st.session_state.page}" width="550" height="900" type="application/pdf"></iframe>
# #     """
# #     with placeholder.container():
# #         html(st.session_state.display_pdf, height=900)
