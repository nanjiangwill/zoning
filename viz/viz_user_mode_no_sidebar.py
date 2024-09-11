import datetime
import glob
import sys
import time
from collections import OrderedDict

import fitz  # PyMuPDF
import orjson as json
import pandas as pd
import streamlit as st
from firebase_admin import exceptions as FirebaseError
from google.api_core.exceptions import GoogleAPIError
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from streamlit_modal import Modal

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

# firestore config
if sys.argv[1]:
    db = firestore.Client.from_service_account_json(sys.argv[1])
else:
    db = firestore.Client.from_service_account_info(
        st.secrets["firebase"]["my_project_settings"]
    )


# Data Loading path
state_experiment_map = {
    "Connecticut": "results/textract_es_gpt4_connecticut_search_range_3",
    "Texas": "results/textract_es_gpt4_texas_search_range_3",
    "North Carolina": "results/textract_es_claude_north_carolina_search_range_3_updated_prompt",
}

pdf_dir_map = {
    "Connecticut": "data/connecticut/pdfs",
    "Texas": "data/texas/pdfs",
    "North Carolina": "data/north_carolina/pdfs",
}

ocr_dir_map = {
    "Connecticut": "data/connecticut/ocr",
    "Texas": "data/texas/ocr",
    "North Carolina": "data/north_carolina/ocr",
}

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

thesarus_file = "data/thesaurus.json"

st.set_page_config(page_title="Zoning", layout="wide")

# Input Name
# Modal for entering the name
modal_name = Modal(
    "Please enter your name to continue:", key="demo-modal", padding=20, max_width=744
)

if (
    "analyst_name" not in st.session_state or not st.session_state["analyst_name"]
) and not modal_name.is_open():
    modal_name.open()

if modal_name.is_open():
    with modal_name.container():
        name_input = st.text_input("Your Name")
        submit_button = st.button("Submit")

        if submit_button:
            if not name_input:
                st.warning("Please enter a valid name")
            else:
                st.session_state["analyst_name"] = name_input
                modal_name.close()

if "analyst_name" not in st.session_state or not st.session_state["analyst_name"]:
    st.write("Please enter valid name to continue")
    st.stop()


# Reading all data
selected_state = "North Carolina"
experiment_dir = state_experiment_map[selected_state]
pdf_dir = pdf_dir_map[selected_state]
ocr_dir = ocr_dir_map[selected_state]

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
all_towns = sorted(list(set([i.place.town for i in all_results["eval"]])))


def get_town_by_place(place: str):
    return Place.from_str(place).town


def filtered_by_place_and_eval(results, place, eval_term):
    return {
        k: [
            result
            for result in results[k]
            if str(result.place) == str(place) and result.eval_term == eval_term
        ]
        for k in results
    }


all_data_by_town = {
    town_name: {
        (eval_term, place): {"place": place, "eval_term": eval_term}
        | filtered_by_place_and_eval(all_results, place, eval_term)
        for place in all_places
        if get_town_by_place(place) == town_name
        for eval_term in all_eval_terms
    }
    for town_name in all_towns
}


def format_town(town_name):
    jstr = " ".join([i[0].upper() + i[1:] for i in town_name.split("-")])
    return f"{jstr}"


format_town_map = {town_name: format_town(town_name) for town_name in all_towns}
inverse_format_town_map = {k: v for v, k in format_town_map.items()}


def get_sorted_eval_district_by_page_first_appeared(all_data_by_town, town_name):
    return sorted(
        (
            (eval_term, town_district)
            for (eval_term, town_district) in all_data_by_town[town_name]
        ),
        key=lambda pair: (
            (
                0,
                all_data_by_town[town_name][pair]["llm"][0]
                .llm_outputs[0]
                .extracted_text[0][1],
            )
            if all_data_by_town[town_name][pair]["llm"][0].llm_outputs[0].extracted_text
            else (
                (
                    1,
                    all_data_by_town[town_name][pair]["search"][
                        0
                    ].entire_search_page_range[0],
                )
                if all_data_by_town[town_name][pair]["search"][
                    0
                ].entire_search_page_range
                else (2, float("inf"))
            )
        ),
    )


# We sort the data by the page their information is first appeared
sorted_all_results = []
for town in all_towns:
    sorted_town_results = get_sorted_eval_district_by_page_first_appeared(
        all_data_by_town, town
    )
    for eval_term, district in sorted_town_results:
        sorted_all_results.append((town, eval_term, district))


# Display the progress bar
def get_firebase_data(selected_state: str, filters: dict = {}) -> pd.DataFrame:
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
    query = doc_ref
    if filters:
        for field, condition in filters.items():
            query = query.where(filter=FieldFilter(field, condition[0], condition[1]))
    docs = query.get()

    # Iterate through documents and extract data
    data = []
    for doc in docs:
        doc_data = doc.to_dict() or {}
        ordered_dict = OrderedDict((k, doc_data.get(k, "")) for k in key_order)
        data.append(ordered_dict)

    sorted_data = sorted(data, key=lambda x: x.get("eval_term", ""))

    # Create a DataFrame from the data
    df = pd.DataFrame(sorted_data)
    return df


def get_next_unlabeled_item(
    labelled_data, sorted_all_results, skip_next_town: bool = False
):
    total_num_items = len(sorted_all_results)
    finished_num_items = len(labelled_data)

    if finished_num_items == 0:
        return (
            total_num_items,
            finished_num_items,
            (
                0,
                sorted_all_results[0][0],
                sorted_all_results[0][1],
                sorted_all_results[0][2],
            ),
        )

    for idx in range(len(sorted_all_results)):
        town, eval_term, district = sorted_all_results[idx]
        # Check if this item has been labeled

        item_labeled = (
            (labelled_data["eval_term"] == format_eval_term[eval_term])
            & (
                labelled_data["district_full_name"]
                == Place.from_str(district).district_full_name
            )
            & (
                labelled_data["district_short_name"]
                == Place.from_str(district).district_short_name
            )
            & (labelled_data["town"] == town)
        ).any()

        if not item_labeled:
            if skip_next_town:
                skip_next_town = False
                continue
            # Return this item, index, and total number of items
            return total_num_items, finished_num_items, (idx, town, eval_term, district)
    # No more items
    return total_num_items, finished_num_items, None


# Skip the data if it's already labeled
try:
    assert st.session_state["analyst_name"]
except KeyError:
    st.write("No data to label without a name")
    st.stop()

labelled_data = get_firebase_data(
    selected_state, {"analyst_name": ["==", st.session_state["analyst_name"]]}
)
total_num_items, finished_num_items, next_item = get_next_unlabeled_item(
    labelled_data, sorted_all_results
)

col1, col2 = st.columns([9, 1])
with col1:
    st.progress(finished_num_items / total_num_items)
with col2:
    st.write(f"Progress: {finished_num_items}/{total_num_items}")


if next_item:
    idx, town_name, eval_term, district = next_item
    st.session_state["current_town"] = town_name
    st.session_state["current_eval_term"] = eval_term
    st.session_state["current_district"] = district
else:
    st.subheader("🎉 You've reached the end of the data!")
    st.download_button(
        label="Download all labeled data (CSV)",
        data=get_firebase_data(
            selected_state, {"analyst_name": ["==", st.session_state["analyst_name"]]}
        ).to_csv(index=True),
        file_name=f"{selected_state}_data.csv",
        mime="text/csv",
    )
    st.stop()


# after reading all data, start the timer
st.session_state["start_time"] = time.time()


# Show the selected data
visualized_data = all_data_by_town[town_name][(eval_term, district)]

place = Place.from_str(district)
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

if llm_output.extracted_text is not None:
    highlight_text_pages = sorted(list(set([i[1] for i in llm_output.extracted_text])))
else:
    highlight_text_pages = []

ground_truth = eval_result.ground_truth
ground_truth_orig = eval_result.ground_truth_orig
ground_truth_page = eval_result.ground_truth_page
answer_correct = eval_result.answer_correct
page_in_range = eval_result.page_in_range

pdf_file = target_pdf(town_name, pdf_dir)
doc = fitz.open(pdf_file)

# Display the title
norm = normalized_llm_output.llm_output.answer

town = "-".join([i[0].upper() + i[1:] for i in place.town.split("-")])


def get_showed_pages(pages, interval):
    showed_pages = []
    for page in pages:
        showed_pages.extend(range(page - interval, page + interval + 1))
    return sorted(list(set(showed_pages)))


showed_pages = get_showed_pages(highlight_text_pages, 1)

if entire_search_page_range == []:
    st.html(
        f"""
        <h2 style="text-align: center; font-size: 2.5em;">
            <em>{format_eval_term[eval_term]}</em> for the
            <em>{place.district_full_name} ({place.district_short_name})</em>
            District in <em>{town}</em>
        </h2>
        <h2><em>Zoning Agent does not find any page related in zoning file</em></h2>
    """
    )

elif len(showed_pages) == 0 and normalized_llm_output.normalized_answer is None:
    st.html(
        f"""
        <h2 style="text-align: center; font-size: 2.5em;">
            <em>{format_eval_term[eval_term]}</em> for the
            <em>{place.district_full_name} ({place.district_short_name})</em>
            District in <em>{town}</em>
        </h2>
        <h2>Value: <em>LLM does not provide an answer</em></h2>
        <h4>Rationale: {llm_output.rationale}</h4>
    """
    )

    to_be_displayed_pages = []
    for display_page_num, display_page in enumerate(entire_search_page_range):
        page = doc.load_page(display_page - 1)
        page_rect = page.rect
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        # Render the page to a PIL Image
        pix = page.get_pixmap(matrix=mat, clip=page_rect)
        img_bytes = pix.pil_tobytes(format="PNG")
        to_be_displayed_pages.append(img_bytes)

    page_img_cols = st.columns(3)
    for k in range(len(to_be_displayed_pages) // 3 + 1):
        for j in range(3):
            i = k * 3 + j
            if i >= len(to_be_displayed_pages):
                continue
            page_img_cols[j].image(
                to_be_displayed_pages[i],
                use_column_width=True,
            )

else:
    st.html(
        f"""
        <h2 style="text-align: center; font-size: 2.5em;">
            <em>{format_eval_term[eval_term]}</em> for the
            <em>{place.district_full_name} ({place.district_short_name})</em>
            District in <em>{town}</em>
        </h2>
        <h2>Value: <em>{norm}</em></h2>
        <h4>Rationale: {llm_output.rationale}</h4>
    """
    )

    if len(showed_pages) == 0:
        showed_pages = entire_search_page_range.copy()

    format_ocr_file = glob.glob(f"{experiment_dir}/format_ocr/{place.town}.json")
    assert len(format_ocr_file) == 1
    format_ocr_file = format_ocr_file[0]
    format_ocr_result = FormatOCR.model_construct(
        **json.loads(open(format_ocr_file).read())
    )

    if "ocr_info" not in st.session_state or not st.session_state["ocr_info"]:
        ocr_file = glob.glob(f"{ocr_dir_map[selected_state]}/{place.town}.json")
        assert len(ocr_file) == 1
        ocr_file = ocr_file[0]
        st.session_state["ocr_info"] = json.loads(open(ocr_file).read())
    extract_blocks = [b for d in st.session_state["ocr_info"] for b in d["Blocks"]]
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
            if i in page_info["text"].lower():
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
                or place.district_short_name.lower() in i[0].lower().split()
            ]
            eval_term_boxs = [
                [i[0], i[1]]
                for i in text_boundingbox
                if any(
                    j.lower() in i[0].lower()
                    for j in expand_term(thesarus_file, eval_term)
                )
            ]
            llm_answer_boxs = [
                [i[0], i[1]]
                for i in text_boundingbox
                if any(j[0].split("\n")[-1] in i[0] for j in llm_output.extracted_text)
            ]  # TODO
            district_color = (1, 0, 0)  # RGB values for red (1,0,0 is full red)
            eval_term_color = (0, 0, 1)  # RGB values for blue (0,0,1 is full blue)
            llm_answer_color = (0, 1, 0)  # RGB values for green (0,1,0 is full green)

            def get_normalized_rect(b):
                if selected_state == "Texas":
                    return fitz.Rect(
                        b["Left"] * page_rect.width,
                        (b["Top"]) * page_rect.height,
                        (b["Left"] + b["Width"]) * page_rect.width,
                        (b["Top"] + b["Height"]) * page_rect.height,
                    )
                elif selected_state == "Connecticut":
                    return fitz.Rect(
                        (1 - b["Top"] - b["Height"]) * page_rect.height,
                        b["Left"] * page_rect.width,
                        (1 - b["Top"]) * page_rect.height,
                        (b["Left"] + b["Width"]) * page_rect.width,
                    )
                elif selected_state == "North Carolina":
                    return fitz.Rect(
                        (b["Left"]) * page_rect.width,
                        (b["Top"]) * page_rect.height,
                        (b["Left"] + b["Width"]) * page_rect.width,
                        (b["Top"] + b["Height"]) * page_rect.height,
                    )
                else:
                    raise ValueError("State not supported")

            def extend_rect(rect):
                # Extend vertically (maintain width, enlarge height to page height)
                vertical_extension = fitz.Rect(rect.x0, 0, rect.x1, page_rect.height)

                # Extend horizontally (maintain height, enlarge width to page width)
                horizontal_extension = fitz.Rect(0, rect.y0, page_rect.width, rect.y1)

                # Combine both extensions
                return (vertical_extension, horizontal_extension)

            def merge_rects(rects):
                if not rects:
                    return []

                merged = [rects[0]]
                for rect in rects[1:]:
                    if any(rect.intersects(m) for m in merged):
                        new_merged = []
                        for m in merged:
                            if rect.intersects(m):
                                rect = rect | m  # Union of rectangles
                            else:
                                new_merged.append(m)
                        new_merged.append(rect)
                        merged = new_merged
                    else:
                        merged.append(rect)
                return merged

            district_rects = [get_normalized_rect(b) for _, b in district_boxs]
            district_rects = merge_rects(district_rects)
            eval_term_rects = [get_normalized_rect(b) for _, b in eval_term_boxs]
            eval_term_rects = merge_rects(eval_term_rects)
            llm_answer_rects = [get_normalized_rect(b) for _, b in llm_answer_boxs]
            llm_answer_rects = merge_rects(llm_answer_rects)

            extended_district_rects = [
                i for rect in district_rects for i in extend_rect(rect)
            ]
            extended_eval_term_rects = [
                i for rect in eval_term_rects for i in extend_rect(rect)
            ]

            overlap_exists = any(
                llm_rect.intersects(rect)
                for llm_rect in llm_answer_rects
                for rect in extended_district_rects + extended_eval_term_rects
            )

            to_be_highlighted_district_rects = []
            to_be_highlighted_eval_term_rects = []
            to_be_highlighted_llm_answer_rects = []
            if overlap_exists:
                for llm_rect in llm_answer_rects:
                    if any(
                        llm_rect.intersects(rect)
                        for rect in extended_district_rects + extended_eval_term_rects
                    ):
                        # Draw only overlapping district and eval term rects
                        overlapping_district_rects = [
                            rect
                            for rect in district_rects
                            if any(llm_rect.intersects(i) for i in extend_rect(rect))
                        ]
                        overlapping_eval_term_rects = [
                            rect
                            for rect in eval_term_rects
                            if any(llm_rect.intersects(i) for i in extend_rect(rect))
                        ]

                        for rect in overlapping_district_rects:
                            to_be_highlighted_district_rects.append([rect, 0.1])
                        for rect in overlapping_eval_term_rects:
                            to_be_highlighted_eval_term_rects.append([rect, 0.1])

                        to_be_highlighted_llm_answer_rects.append([llm_rect, 0.5])
            else:
                to_be_highlighted_district_rects = [
                    [rect, 0.15] for rect in district_rects
                ]
                to_be_highlighted_eval_term_rects = [
                    [rect, 0.15] for rect in eval_term_rects
                ]
                to_be_highlighted_llm_answer_rects = [
                    [rect, 0.15] for rect in llm_answer_rects
                ]

            for rect, opacity in to_be_highlighted_district_rects:
                page.draw_rect(
                    rect,
                    fill=district_color,
                    width=1,
                    stroke_opacity=0,
                    fill_opacity=opacity,
                )

            for rect, opacity in to_be_highlighted_eval_term_rects:
                page.draw_rect(
                    rect,
                    fill=eval_term_color,
                    width=1,
                    stroke_opacity=0,
                    fill_opacity=opacity,
                )

            for rect, opacity in to_be_highlighted_llm_answer_rects:
                page.draw_rect(
                    rect,
                    fill=llm_answer_color,
                    width=1,
                    stroke_opacity=0,
                    fill_opacity=opacity,
                )

        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        # Render the page to a PIL Image
        pix = page.get_pixmap(matrix=mat, clip=page_rect)
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
                use_column_width=True,
            )

st.divider()


# write data
def write_data(human_feedback: str) -> bool:
    town_name = place.town
    district_full_name = place.district_full_name
    district_short_name = place.district_short_name

    # Store and reset the timer
    if "start_time" not in st.session_state:
        elapsed_sec = -1
    else:
        elapsed_sec = time.time() - st.session_state["start_time"]

    try:
        assert "analyst_name" in st.session_state and st.session_state["analyst_name"]
    except KeyError:
        st.toast("Please enter your name first", icon="🚨")
        return False

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

    try:
        doc_ref = db.collection(selected_state)
        _, result = doc_ref.add(d)
        result.get()  # Wait for acknowledgment
        st.session_state["start_time"] = time.time()  # Reset the timer

        st.toast("Going to next data in 2 seconds", icon="🚀")
        st.toast("Data successfully written to database!", icon="🎉")

        return True

    except GoogleAPIError as e:
        st.error(f"Error writing to Firestore: {e}")
        return False
    except FirebaseError as e:
        st.error(f"Firebase SDK Error: {e}")
        return False


_, _, next_next_item = get_next_unlabeled_item(
    labelled_data, sorted_all_results, skip_next_town=True
)


def jump_to_next_one():
    if next_next_item:
        idx, next_town_name, next_eval_term, next_district = next_next_item
        if st.session_state["current_town"] != next_town_name:
            st.session_state["ocr_info"] = None  # Reset the OCR info
            st.session_state["finish-town-opened"] = True
            st.session_state["model_bext_town_text"] = (
                f"🎉 You've finished the data for {town_name}!\n"
                f"Next town: {next_town_name}."
            )
    else:
        st.subheader("🎉 You've reached the end of the data!")
        st.stop()


# Post writing data
model_next_town = Modal("", key="finish-town", padding=20, max_width=744)
if "finish-town-opened" not in st.session_state:
    st.session_state["finish-town-opened"] = False
if st.session_state["finish-town-opened"]:
    with model_next_town.container():
        st.header(st.session_state["model_bext_town_text"])

with st.container(border=True):
    correct_col, not_sure_col, wrong_col = st.columns(3)

    def button_callback(feedback):
        def _button_callback():
            if write_data(feedback):
                jump_to_next_one()

        return _button_callback

    with correct_col:
        st.button(
            "Verified Correct",
            key="llm_correct",
            type="primary",
            use_container_width=True,
            on_click=button_callback("correct"),
        )

    with not_sure_col:
        st.button(
            "Not Enough Information",
            key="llm_not_sure",
            type="secondary",
            use_container_width=True,
            on_click=button_callback("not_sure"),
        )

    with wrong_col:
        st.button(
            "Verified Incorrect",
            key="llm_wrong",
            type="secondary",
            use_container_width=True,
            on_click=button_callback("wrong"),
        )


# Display the next item

if next_next_item:
    idx, next_town_name, next_eval_term, next_district = next_next_item
    next_place = Place.from_str(next_district)
    st.html(
        f"""
        <h2 style="text-align: center; font-size: 2.5em;">
            <em>Next item: {format_eval_term[next_eval_term]}</em> for the
            <em>{next_place.district_full_name} ({next_place.district_short_name})</em>
            District in <em>{format_town_map[next_town_name]}</em>
        </h2>
    """
    )
else:
    st.write("No more items to label")


st.link_button(f"PDF Link for {format_town_map[town_name]}", pdf_file)

st.download_button(
    label="Download all labeled data (CSV)",
    data=get_firebase_data(
        selected_state,
        filters={"analyst_name": ["==", st.session_state["analyst_name"]]},
    ).to_csv(index=True),
    file_name=f"{selected_state}_data.csv",
    mime="text/csv",
)
