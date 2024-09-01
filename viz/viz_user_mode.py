import datetime
import glob
import sys
import time
from collections import OrderedDict

import fitz  # PyMuPDF

# import json
import orjson as json
import pandas as pd
import requests
import streamlit as st
from streamlit_modal import Modal
from google.cloud import firestore

from zoning.class_types import (
    EvalResult,
    FormatOCR,
    LLMInferenceResult,
    NormalizedLLMInferenceResult,
    Place,
    PromptResult,
    SearchResult,
)
from zoning.utils import expand_term, target_pdf

if sys.argv[1]:
    db = firestore.Client.from_service_account_json(sys.argv[1])
else:
    db = firestore.Client.from_service_account_info(
        st.secrets["firebase"]["my_project_settings"]
    )


state_experiment_map = {
    "Connecticut": "results/textract_es_gpt4_connecticut_search_range_3",
    "Texas": "results/textract_es_gpt4_texas_search_range_3",
    "North Carolina": "results/textract_es_gpt4_north_carolina_search_range_3_old_thesaurus",
}

st.set_page_config(page_title="Zoning", layout="wide")

thesarus_file = "data/thesaurus.json"

format_eval_term = {
    "floor_to_area_ratio": "Floor to Area Ratio",
    "max_height": "Max Height",
    "max_lot_coverage": "Max Lot Coverage",
    "max_lot_coverage_pavement": "Max Lot Coverage Pavement",
    "min_lot_size": "Min Lot Size",
    "min_parking_spaces": "Min Parking Spaces",
    "min_unit_size": "Min Unit Size",
}

inverse_format_eval_term = {k: v for v, k in format_eval_term.items()}

# Modal for entering the name
modal = Modal(
    "Please enter your name to continue:",
    key="demo-modal",

    padding=20,
    max_width=744
)


def write_data(human_feedback: str):
    if "analyst_name" not in st.session_state or not st.session_state["analyst_name"]:
        modal.open()
    else:
        town_name = place.town
        district_full_name = place.district_full_name
        district_short_name = place.district_short_name

        # Store and reset the timer
        if "start_time" not in st.session_state:
            elapsed_sec = -1
        else:
            elapsed_sec = time.time() - st.session_state["start_time"]
        st.session_state["start_time"] = time.time()

        d = {
            "analyst_name": st.session_state["analyst_name"],
            "state": selected_state,
            "town": town_name,
            "district_full_name": district_full_name,
            "district_short_name": district_short_name,
            "eval_term": format_eval_term[eval_term],
            "human_feedback": human_feedback,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": elapsed_sec,
        }

        doc_ref = db.collection(selected_state)

        doc_ref.add(d)

        st.toast("Going to next data in 2 seconds", icon="🚀")
        st.toast("Data successfully written to database!", icon="🎉")


def get_firebase_csv_data(selected_state: str):
    key_order = [
        "eval_term",
        "state",
        "town",
        "district_full_name",
        "district_short_name",
        "human_feedback",
        "analyst_name",
        "date",
    ]

    doc_ref = db.collection(selected_state)

    docs = doc_ref.get()

    data = []

    # Iterate through documents and extract data
    for doc in docs:
        ordered_dict = OrderedDict((k, doc.to_dict().get(k, "")) for k in key_order)
        data.append(ordered_dict)

    sorted_data = sorted(data, key=lambda x: x.get("eval_term", ""))

    # Create a DataFrame from the data
    df = pd.DataFrame(sorted_data)

    return df.to_csv(index=True)

# Initialize session state variables
if "analyst_name" not in st.session_state:
    st.session_state["analyst_name"] = ""

if not st.session_state["analyst_name"] and not modal.is_open():
    modal.open()

if modal.is_open():
    with modal.container():
        name_input = st.text_input("Your Name")
        submit_button = st.button("Submit")

        if submit_button:
            if not name_input:
                st.warning("Please enter a valid name")
            else:
                st.session_state["analyst_name"] = name_input
                st.session_state["start_time"] = time.time()
                modal.close()

if "analyst_name" in st.session_state and st.session_state["analyst_name"]:
    st.sidebar.subheader(f"Hello, {st.session_state['analyst_name']}!")

# Sidebar config
with st.sidebar:
    # Step 0: load files

    selected_state = st.selectbox(
        "Select a state",
        [
            "North Carolina",
            "Connecticut",
            "Texas",
        ],
        index=0,
    )

    def format_state(state):
        return state.lower().replace(" ", "_")

    experiment_dir = state_experiment_map[selected_state]
    # s3_pdf_dir = f"https://zoning-nan.s3.us-east-2.amazonaws.com/pdf/{format_state(selected_state)}"
    pdf_dir = f"{experiment_dir}/pdf"

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

    def show_fullname_shortname(place):
        town, district_short_name, district_full_name = place.split("__")
        jstr = "-".join([i[0].upper() + i[1:] for i in town.split("-")])
        return f"{jstr}, {district_full_name} ({district_short_name})"

    format_place_map = {place: show_fullname_shortname(place) for place in all_places}

    inverse_format_place_map = {k: v for v, k in format_place_map.items()}

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

    # st.write(
    #     "Number of :blue-background[all] selected data: ",
    #     len(all_places),
    # )

    # Step 1: Config
    st.divider()
    st.subheader("Step 1: Select Field", divider="rainbow")

    eval_term = st.radio(
        "All available fields",
        # [format_eval_term[i] for i in all_eval_terms],
        [
            format_eval_term[i]
            for i in [
                "max_height",
                "max_lot_coverage",
                "min_lot_size",
                "min_parking_spaces",
            ]
        ],
        index=0,
        # key="eval_term",
    )
    eval_term = inverse_format_eval_term[eval_term]

    selected_data = [i for _, i in sorted(all_data_by_eval_term[eval_term].items())]

    # Step 2: Select one data to check
    st.divider()
    st.subheader("Step 2: Select one district to check", divider="rainbow")

    place = st.radio(
        "All available districts",
        (
            format_place_map[term["place"]]
            for term in selected_data
            if str(term["place"])
            not in [
                "bethel__MR__Multi-family Residential",
                "canton__C-4__Highway business",
                "harmony__R-O__Residential Office",
                "falkland__A-R__Agricultural-Residential",
                "knightdale__HB__Highway Business",
            ]
        ),
        # key="place",
        index=0,
    )

    st.subheader("Step 3: Download all labeled data", divider="rainbow")
    st.download_button(
        label="Download CSV",
        data=get_firebase_csv_data(selected_state),
        file_name=f"{selected_state}_data.csv",
        mime="text/csv",
    )

# Load the data for the town.
place = inverse_format_place_map[place]

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

pdf_file = target_pdf(place.town, pdf_dir)

doc = fitz.open(pdf_file)

norm = normalized_llm_output.normalized_answer
if isinstance(norm, list):
    norm = norm[0]

town = "-".join([i[0].upper() + i[1:] for i in place.town.split("-")])
st.html(
    f"""
    <h2><em>{format_eval_term[eval_term]}</em> for the <em>{place.district_full_name}({place.district_short_name})</em> District in <em>{town}</em></h2>
    <h2>Value: <em>{norm}</em></h2>
    <h4>Rationale: {llm_output.rationale}</h4>
"""
)


def get_showed_pages(pages, interval):
    showed_pages = []
    for page in pages:
        showed_pages.extend(range(page - interval, page + interval + 1))
    return sorted(list(set(showed_pages)))


showed_pages = get_showed_pages(highlight_text_pages, 1)

if len(showed_pages) == 0 and normalized_llm_output.normalized_answer == None:
    st.write("LLM does not find any page related")

else:
    if len(showed_pages) == 0:
        showed_pages = entire_search_page_range.copy()

    format_ocr_file = glob.glob(f"{experiment_dir}/format_ocr/{place.town}.json")
    assert len(format_ocr_file) == 1
    format_ocr_file = format_ocr_file[0]
    format_ocr_result = FormatOCR.model_construct(
        **json.loads(open(format_ocr_file).read())
    )

    ocr_file = glob.glob(f"{experiment_dir}/ocr/{place.town}.json")
    assert len(ocr_file) == 1
    ocr_file = ocr_file[0]
    ocr_info = json.loads(open(ocr_file).read())

    extract_blocks = [b for d in ocr_info for b in d["Blocks"]]
    edited_pages = []
    for shown_page_num, show_page in enumerate(showed_pages):
        page = doc.load_page(show_page - 1)
        page_rect = page.rect
        # for zoom in
        page_info = [i for i in format_ocr_result.pages if i["page"] == str(show_page)]
        assert len(page_info) == 1
        page_info = page_info[0]

        load_ocr = False
        for i in expand_term(thesarus_file, eval_term):
            if i in page_info["text"]:
                load_ocr = True
        if (
            place.town.lower() in page_info["text"].lower()
            or place.district_full_name.lower() in page_info["text"].lower()
            or place.district_short_name.lower() in page_info["text"].lower()
        ):
            load_ocr = True

        if load_ocr:
            page_ocr_info = [w for w in extract_blocks if w["Page"] == show_page]
            text_boundingbox = [
                (w["Text"], w["Geometry"]["BoundingBox"])
                for w in page_ocr_info
                if "Text" in w
            ]
            district_boxs = [
                [i[0], i[1]]
                for i in text_boundingbox
                if place.district_full_name.lower() in i[0].lower()
                or place.district_short_name in i[0]
            ]
            eval_term_boxs = [
                [i[0], i[1]]
                for i in text_boundingbox
                if any(j in i[0] for j in expand_term(thesarus_file, eval_term))
            ]
            llm_answer_boxs = [
                [i[0], i[1]]
                for i in text_boundingbox
                if any(j.split("\n")[-1] in i[0] for j in llm_output.extracted_text)
            ]  # TODO

            district_color = (1, 0, 0)  # RGB values for red (1,0,0 is full red)
            eval_term_color = (0, 0, 1)  # RGB values for blue (0,0,1 is full blue)
            llm_answer_color = (0, 1, 0)  # RGB values for green (0,1,0 is full green)

            box_list = [district_boxs, eval_term_boxs, llm_answer_boxs]
            color_list = [district_color, eval_term_color, llm_answer_color]

            for box, color in zip(box_list, color_list):
                for _, b in box:
                    # b["Left"] += 50
                    # b["Top"] += 50
                    # b["Width"] -= 50
                    # b["Height"] -= 50
                    if selected_state == "Texas":
                        normalized_rect = fitz.Rect(
                            b["Left"] * page_rect.width,
                            (b["Top"]) * page_rect.height,
                            (b["Left"] + b["Width"]) * page_rect.width,
                            (b["Top"] + b["Height"]) * page_rect.height,
                        )
                    elif selected_state == "Connecticut":
                        normalized_rect = fitz.Rect(
                            (1 - b["Top"] - b["Height"]) * page_rect.height,
                            b["Left"] * page_rect.width,
                            (1 - b["Top"]) * page_rect.height,
                            (b["Left"] + b["Width"]) * page_rect.width,
                        )
                    elif selected_state == "North Carolina":
                        normalized_rect = fitz.Rect(
                            (b["Left"]) * page_rect.width,
                            (b["Top"]) * page_rect.height,
                            (b["Left"] + b["Width"]) * page_rect.width,
                            (b["Top"] + b["Height"]) * page_rect.height,
                        )
                    else:
                        raise ValueError("State not supported")
                    page.draw_rect(
                        normalized_rect,
                        fill=color,
                        width=1,
                        stroke_opacity=0,
                        fill_opacity=0.15,
                    )

        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        # Render the page to a PIL Image
        pix = page.get_pixmap(matrix=mat, clip=page_rect)
        # pix = page.get_pixmap()
        img_bytes = pix.pil_tobytes(format="PNG")
        edited_pages.append(img_bytes)

    page_img_cols = st.columns(3)

    for k in range(len(showed_pages) // 3 + 1):
        for j in range(3):
            i = k * 3 + j
            if i >= len(showed_pages):
                continue
            page_img_cols[j].image(
                edited_pages[i],
                # caption=f"Page {showed_pages[i]}",
                use_column_width=True,
                # width=400
            )


st.divider()
with st.container(border=True):
    # st.subheader("Current data")
    correct_col, not_sure_col, wrong_col = st.columns(3)
    with correct_col:
        if st.button(
            "Verified Correct",
            key="llm_correct",
            type="primary",
            use_container_width=True,
        ):
            # print(place)
            write_data("correct")


    with not_sure_col:
        if st.button(
            "Not Enough Information",
            key="llm_not_sure",
            type="secondary",
            use_container_width=True,
        ):
            write_data("not_sure")
    with wrong_col:
        if st.button(
            "Verified Incorrect",
            key="llm_wrong",
            type="secondary",
            use_container_width=True,
        ):
            write_data("wrong")
st.link_button("PDF Link", pdf_file)
