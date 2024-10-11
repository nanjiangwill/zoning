import datetime
import glob
import os
import time
from collections import OrderedDict

import fitz  # PyMuPDF
import orjson as json
import pandas as pd
import requests
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
from zoning.utils import expand_term

# firestore config

db = firestore.Client.from_service_account_info(
    st.secrets["firebase"]["my_project_settings"]
)

# Data Loading path
state_experiment_map = {
    "North Carolina": "results/textract_es_claude_north_carolina_search_range_3_updated_prompt",
}

# pdf_dir_map = {
#     "Connecticut": "data/connecticut/pdfs",
#     "Texas": "data/texas/pdfs",
#     "North Carolina": "data/north_carolina/pdfs",
# }

# ocr_dir_map = {
#     "Connecticut": "data/connecticut/ocr",
#     "Texas": "data/texas/ocr",
#     "North Carolina": "data/north_carolina/ocr",
# }

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
    "Zoning Agent Instructions", key="demo-modal", padding=20, max_width=744
)

if (
        "analyst_name" not in st.session_state or not st.session_state["analyst_name"]
) and not modal_name.is_open():
    modal_name.open()

if modal_name.is_open():
    with modal_name.container():
        with st.chat_message("system", avatar="🖥️"):
            st.subheader("How to use the Zoning Agent")
            st.write(
                """
            1. Login with your name and click on the "Start" button
            2. Read title for item carefully, it contains the current eval term and the district name
            3. You will find the LLM answer and related PDF pages below to help you with the labeling decision
            3.1. Information will be downloaded automatically when you click on the "Start" button
            4. There will be highlights on the PDF page to help you with the labeling decision
            5. After carefully reviewing the data, you need to click\n
            • `"Verified Correct"` if the LLM answer is correct\n
            • `"Verified Incorrect"` if the LLM answer is incorrect\n
            • `"Not Enough Information"` if you are not sure about the answer\n
            6. It will automatically jump to the next item
            7. Gather feedback to help improve the Zoning Agent!
            8. You can download the labeled data by clicking the "Download all labeled data (CSV)" button
            9. You can leave any time and resume later
            """
            )
            st.subheader("Meaning of the highlights")
            st.write(
                """
            1. :red[Red] highlights indicate the district of the zoning regulation.
            2. :blue[Blue] highlights indicate the eval term of the zoning regulation.
            3. :green[Green] highlights indicate the LLM answer of the zoning regulation.
            """
            )

        name_input = st.text_input("Your Name")
        submit_button = st.button("Start")
        if submit_button:
            if not name_input:
                st.warning("Please enter a valid name")
            else:
                st.session_state["analyst_name"] = name_input
                modal_name.close()

if "analyst_name" not in st.session_state or not st.session_state["analyst_name"]:
    st.write("Please enter valid name to continue")
    st.stop()


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


# all_data_by_town = {
#     town_name: {
#         (eval_term, place): {"place": place, "eval_term": eval_term}
#         | filtered_by_place_and_eval(all_results, place, eval_term)
#         for place in all_places
#         if get_town_by_place(place) == town_name
#         for eval_term in all_eval_terms
#     }
#     for town_name in all_towns
# }


def format_town(town_name):
    jstr = " ".join([i[0].upper() + i[1:] for i in town_name.split("-")])
    return f"{jstr}"


# def get_sorted_eval_district_by_page_first_appeared(all_data_by_town, town_name):
#     return sorted(
#         (
#             (eval_term, town_district)
#             for (eval_term, town_district) in all_data_by_town[town_name]
#         ),
#         key=lambda pair: (
#             (
#                 0,
#                 all_data_by_town[town_name][pair]["llm"][0]
#                 .llm_outputs[0]
#                 .extracted_text[0][1],
#             )
#             if all_data_by_town[town_name][pair]["llm"] and all_data_by_town[town_name][pair]["llm"][0].llm_outputs and all_data_by_town[town_name][pair]["llm"][0].llm_outputs[0].extracted_text
#             else (
#                 (
#                     1,
#                     all_data_by_town[town_name][pair]["search"][
#                         0
#                     ].entire_search_page_range[0],
#                 )
#                 if all_data_by_town[town_name][pair]["search"] and all_data_by_town[town_name][pair]["search"][
#                     0
#                 ].entire_search_page_range
#                 else (2, float("inf"))
#             )
#         ),
#     )


# # We sort the data by the page their information is first appeared
# sorted_all_results = []
# for town in all_towns:
#     sorted_town_results = get_sorted_eval_district_by_page_first_appeared(
#         all_data_by_town, town
#     )
#     for eval_term, district in sorted_town_results:
#         if (
#             all_data_by_town[town][(eval_term, district)]["llm"] and
#             all_data_by_town[town][(eval_term, district)]["llm"][0].llm_outputs and
#             all_data_by_town[town][(eval_term, district)]["llm"][0]
#             .llm_outputs[0]
#             .extracted_text
#             is not None
#         ):
#             sorted_all_results.append((town, eval_term, district))


# Display the progress bar
def get_firebase_data(selected_state: str, filters: dict = {}) -> pd.DataFrame:
    key_order = [
        "eval_term",
        "state",
        "town",
        "district_full_name",
        "district_short_name",
        "llm_answer",
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


def prepare_data_for_download(selected_state: str, filters: dict = {}):
    all_labelled_data = get_firebase_data(selected_state, filters)
    # Group the data by district (combining full name and short name)
    grouped_data = all_labelled_data.groupby(
        ["state", "town", "district_full_name", "district_short_name"]
    )

    # Create a list to store the merged data
    merged_data = []

    # Iterate through each group
    for (state, town, district_full, district_short), group in grouped_data:
        # Create a dictionary to store the merged row
        merged_row = {
            "State": state,
            "Town": format_town(town),
            "District Full Name": district_full,
            "District Short Name": district_short,
        }

        # Iterate through each row in the group
        for _, row in group.iterrows():
            # Add the eval_term as a column, with its llm_answer and human_feedback as the values
            merged_row[f"{row['eval_term']} LLM Answer"] = row["llm_answer"]
            merged_row[f"{row['eval_term']} Human Feedback"] = row["human_feedback"]

        # Add analyst name and date (assuming these are the same for all rows in a group)
        merged_row["Analyst Name"] = group["analyst_name"].iloc[0]
        merged_row["Date"] = group["date"].iloc[0]

        # Append the merged row to our list
        merged_data.append(merged_row)

    # Convert the merged data back to a DataFrame
    merged_df = pd.DataFrame(merged_data)

    # Reorder columns to have eval terms after district information
    eval_terms = [col for col in merged_df.columns if col in format_eval_term.values()]
    other_cols = [col for col in merged_df.columns if col not in eval_terms]
    column_order = other_cols[:4] + eval_terms + other_cols[4:]

    merged_df = merged_df[column_order]

    return merged_df


def download_file_with_progress(url):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 10 * 1024 * 1024  # 5 MB

    progress_bar = st.progress(0)
    progress_text = st.empty()

    data = b""
    for data_chunk in response.iter_content(block_size):
        data += data_chunk
        progress = len(data) / total_size
        progress_bar.progress(progress)
        progress_text.text(
            f"Downloaded: {len(data) / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB (Will only download once for one town)"
        )

    return data


def build_batches_for_town(town_name, all_data_by_town):
    town_data = all_data_by_town[town_name]

    combinations_with_answer = []
    combinations_without_answer = []

    for (eval_term, district), data in town_data.items():
        try:
            llm_output = data['llm'][0].llm_outputs[0]
        except:
            continue

        if llm_output.extracted_text is not None:
            # Get the page number where the information first appears
            first_page = llm_output.extracted_text[0][1]
            combinations_with_answer.append({
                'eval_term': eval_term,
                'district': district,
                'first_page': first_page,
                'town_name': town_name,
            })
        else:
            # RAG did not find an answer
            combinations_without_answer.append({
                'eval_term': eval_term,
                'district': district,
                'town_name': town_name,
            })

    # Sort combinations with answer by first_page
    combinations_with_answer.sort(key=lambda x: x['first_page'])

    # Group combinations into batches where the page numbers are within a certain gap
    batches_with_answer = []
    current_batch = []
    max_page_gap = 1  # Adjust as needed

    for combo in combinations_with_answer:
        if not current_batch:
            current_batch.append(combo)
        else:
            if combo['first_page'] - current_batch[-1]['first_page'] <= max_page_gap:
                current_batch.append(combo)
            else:
                # Sort current batch by eval_term before adding
                current_batch.sort(key=lambda x: x['eval_term'])
                batches_with_answer.append(current_batch)
                current_batch = [combo]

    if current_batch:
        current_batch.sort(key=lambda x: x['eval_term'])
        batches_with_answer.append(current_batch)

    # Group combinations without answer by eval_term
    batches_without_answer = []
    eval_terms = sorted(set(c['eval_term'] for c in combinations_without_answer))
    for eval_term in eval_terms:
        batch = [c for c in combinations_without_answer if c['eval_term'] == eval_term]
        # Sort batch by district or any other criteria if needed
        batch.sort(key=lambda x: x['district'])
        batches_without_answer.append(batch)

    # Combine batches
    batches = batches_with_answer + batches_without_answer

    return batches


def get_next_unlabeled_batch(labelled_data, all_batches):
    finished_num_batches = 0

    for idx, batch in enumerate(all_batches):
        batch_labelled = False
        for item in batch:
            item_labeled = (
                (labelled_data["eval_term"] == format_eval_term[item['eval_term']])
                & (
                    labelled_data["district_full_name"]
                    == Place.from_str(item['district']).district_full_name
                )
                & (
                    labelled_data["district_short_name"]
                    == Place.from_str(item['district']).district_short_name
                )
                & (labelled_data["town"] == item['town_name'])
            ).any()
            if item_labeled:
                batch_labelled = True
                break
        if not batch_labelled:
            # Return this batch
            return len(all_batches), finished_num_batches, (idx, batch)
        else:
            finished_num_batches += 1
    # No more batches
    return len(all_batches), finished_num_batches, None


# Reading all data
selected_state = "North Carolina"
batched_data_path = "results/textract_es_claude_north_carolina_search_range_3_updated_prompt/sorted_all_results_with_search_batched.json"

if os.path.exists(batched_data_path):
    # Load the precomputed batches
    all_batches = json.loads(open(batched_data_path).read())
else:
    experiment_dir = state_experiment_map[selected_state]

    # Load all results by reading individual JSON files from their respective directories
    all_results = {
        k: [
            X.model_construct(**json.loads(open(i, 'r').read()))
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

    # Extract unique evaluation terms, places, and towns
    all_eval_terms = sorted(list(set([i.eval_term for i in all_results["eval"]])))
    all_places = sorted(list(set(str(i.place) for i in all_results["eval"])))
    all_towns = sorted(list(set([i.place.town for i in all_results["eval"]])))

    # Organize data by town
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

    # Build all batches
    all_batches = []
    for town in all_towns:
        town_batches = build_batches_for_town(town, all_data_by_town)
        all_batches.extend(town_batches)
    print(all_batches)

    # Save the batched data for future runs
    json_bytes = json.dumps(all_batches)

    # Write the bytes to a file
    with open(batched_data_path, 'wb') as f:
        f.write(json_bytes)


all_towns = sorted(list(set([i['town_name'] for batch in all_batches for i in batch])))
format_town_map = {town_name: format_town(town_name) for town_name in all_towns}
inverse_format_town_map = {k: v for v, k in format_town_map.items()}

# Skip the data if it's already labeled
try:
    assert st.session_state["analyst_name"]
except KeyError:
    st.write("No data to label without a name")
    st.stop()

labelled_data = get_firebase_data(
    selected_state, {"analyst_name": ["==", st.session_state["analyst_name"]]}
)

total_num_batches, finished_num_batches, next_batch_info = get_next_unlabeled_batch(
    labelled_data, all_batches
)

# Display the progress bar
col1, col2 = st.columns([9, 1])
with col1:
    st.progress(finished_num_batches / total_num_batches)
with col2:
    st.write(f"Progress: {finished_num_batches}/{total_num_batches}")

if next_batch_info:
    idx, batch = next_batch_info
    st.session_state["current_batch"] = batch
    st.session_state["current_batch_index"] = idx  # Store current batch index
else:
    st.subheader("🎉 You've reached the end of the data!")
    st.download_button(
        label="Download all labeled data (CSV)",
        data=prepare_data_for_download(
            selected_state, {"analyst_name": ["==", st.session_state["analyst_name"]]}
        ).to_csv(index=False),
        file_name=f"{selected_state}_data.csv",
        mime="text/csv",
    )
    st.stop()

# after reading all data, start the timer
st.session_state["start_time"] = time.time()

# Display the batch information
batch = st.session_state["current_batch"]
town_name = batch[0]['town_name']
st.session_state["current_town"] = town_name

# For visualized data
s3_prefix = "https://zoning-nan.s3.us-east-2.amazonaws.com/results/north_carolina_claude"

# Initialize sets to collect pages and highlights
all_showed_pages = set()
all_highlight_info = []

with st.sidebar:
    st.markdown(f"# Town: {format_town(town_name)}")

    for item in batch:
        eval_term = item['eval_term']
        district = item['district']
        place = Place.from_str(item['district'])

        current_viewing_data_name = f"{item['eval_term']}__{item['district'].replace(' ', '+')}.json"
        visualized_data = {
            k: [
                X.model_construct(**json.loads(requests.get(f"{s3_prefix}/{k}/{current_viewing_data_name}").text))
            ]
            for k, X in [
                ("search", SearchResult),
                ("prompt", PromptResult),
                ("llm", LLMInferenceResult),
                ("normalization", NormalizedLLMInferenceResult),
                ("eval", EvalResult),
            ]
        }


        # loading info
        search_result = visualized_data["search"][0]
        entire_search_page_range = search_result.entire_search_page_range
        llm_inference_result = visualized_data["llm"][0]
        llm_output = llm_inference_result.llm_outputs[0]
        normalized_llm_inference_result = visualized_data["normalization"][0]
        normalized_llm_output = normalized_llm_inference_result.normalized_llm_outputs[0]
        norm = normalized_llm_output.llm_output.answer
        eval_result = visualized_data["eval"][0]

        town_formatted = format_town(town_name)


        # Collect pages to display
        def get_showed_pages(pages, interval):
            showed_pages = []
            for page in pages:
                showed_pages.extend(range(page - interval, page + interval + 1))
            return sorted(list(set(showed_pages)))


        if llm_output.extracted_text is not None:
            highlight_text_pages = sorted(list(set([i[1] for i in llm_output.extracted_text])))
        else:
            highlight_text_pages = []

        # Collect pages to display
        showed_pages = get_showed_pages(highlight_text_pages, 1)
        if len(showed_pages) == 0:
            showed_pages = entire_search_page_range.copy()

        all_showed_pages.update(showed_pages)

        # Collect highlight info
        all_highlight_info.append({
            'eval_term': eval_term,
            'district': district,
            'place': place,
            'llm_output': llm_output,
        })

        # Display the title (result item)
        if entire_search_page_range == []:
            st.html(
                f"""
                    <div style="border: 1px solid #ccc; padding: 10px; margin: 5px; border-radius: 5px;">
                        <p style="font-size: 1.2em;">
                            <b><em>{format_eval_term[eval_term]}</em> / {place.district_full_name} ({place.district_short_name})</b>
                        </p>
                        <p style="font-size: 1.2em;"><b>Value: <em>Zoning Agent does not find any page related in zoning file</em></b></p>
                    </div>
                """
            )
        elif len(highlight_text_pages) == 0 and normalized_llm_output.normalized_answer is None:
            st.html(
                f"""
                    <div style="border: 1px solid #ccc; padding: 10px; margin: 5px; border-radius: 5px;">
                        <p style="font-size: 1.2em;">
                            <b><em>{format_eval_term[eval_term]}</em> / {place.district_full_name} ({place.district_short_name})</b>
                        </p>
                        <p style="font-size: 1.2em;"><b>Value: <em>LLM does not provide an answer</em></b></p>
                        <p>Rationale: {llm_output.rationale}</p>
                    </div>
                """
            )
        else:
            st.html(
                f"""
                    <div style="border: 1px solid #ccc; padding: 10px; margin: 5px; border-radius: 5px;">
                        <p style="font-size: 1.2em;">
                            <b>
                            {format_eval_term[eval_term]} /
                            {place.district_full_name} ({place.district_short_name})
                            </b>
                        </p>
                        <p style="font-size: 1.2em;"><b>Value: <em>{norm}</em></b></p>
                        <p>Rationale: {llm_output.rationale}</p>
                    </div>
                """
            )

        # ground_truth = eval_result.ground_truth
        # ground_truth_orig = eval_result.ground_truth_orig
        # ground_truth_page = eval_result.ground_truth_page
        # answer_correct = eval_result.answer_correct
        # page_in_range = eval_result.page_in_range

def get_edited_pages(all_showed_pages, all_highlight_info, format_ocr_result, extract_blocks, llm_output, place, eval_term, selected_state):

    def get_normalized_rect(b, page_rect):
        if selected_state == "Texas":
            return fitz.Rect(
                b["Left"] * page_rect.width,
                b["Top"] * page_rect.height,
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
                b["Left"] * page_rect.width,
                b["Top"] * page_rect.height,
                (b["Left"] + b["Width"]) * page_rect.width,
                (b["Top"] + b["Height"]) * page_rect.height,
            )
        else:
            raise ValueError("State not supported")

    def extend_rect(rect, page_rect):
        # Extend vertically
        vertical_extension = fitz.Rect(rect.x0, 0, rect.x1, page_rect.height)
        # Extend horizontally
        horizontal_extension = fitz.Rect(0, rect.y0, page_rect.width, rect.y1)
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

    edited_pages = []
    pdfdata = st.session_state["doc"].tobytes()
    temp = fitz.open("pdf", pdfdata)
    for shown_page_num, show_page in enumerate(all_showed_pages):
        # Load the page and create a copy
        page = temp.load_page(show_page - 1)
        page_rect = page.rect
        # for zoom in
        page_info = [i for i in format_ocr_result.pages if i["page"] == str(show_page)]
        assert len(page_info) == 1
        page_info = page_info[0]

        # Decide whether to load OCR for this page
        load_ocr = False
        page_text_lower = page_info["text"].lower()
        for item in all_highlight_info:
            eval_term = item['eval_term']
            place = item['place']
            for term in expand_term(thesarus_file, eval_term):
                if term in page_text_lower:
                    load_ocr = True
                    break
            if (
                    place.town.lower() in page_text_lower
                    or place.district_full_name.lower() in page_text_lower
                    or place.district_short_name.lower() in page_text_lower
            ):
                load_ocr = True
        if load_ocr:
            # Get OCR info for the page
            page_ocr_info = [w for w in extract_blocks if w["Page"] == show_page]
            text_boundingbox = [
                (w["Text"], w["Geometry"]["BoundingBox"])
                for w in page_ocr_info
                if "Text" in w
            ]

            # Initialize lists for rectangles
            district_rects = []
            eval_term_rects = []
            llm_answer_rects = []

            # Apply highlights per item
            for item in all_highlight_info:
                eval_term = item['eval_term']
                place = item['place']
                llm_output = item['llm_output']

                # Identify district boxes
                district_boxes = [
                    [i[0], i[1]]
                    for i in text_boundingbox
                    if place.district_full_name.lower() in i[0].lower()
                       or place.district_full_name.lower() in " ".join(i[0].lower().split())
                       or place.district_short_name.lower() in i[0].lower().split()
                ]
                district_rects.extend([get_normalized_rect(b[1], page_rect) for b in district_boxes])

                # Identify eval_term boxes
                eval_term_boxes = [
                    [i[0], i[1]]
                    for i in text_boundingbox
                    if any(
                        term.lower() in " ".join(i[0].lower().split())
                        for term in expand_term(thesarus_file, eval_term)
                    )
                ]
                eval_term_rects.extend([get_normalized_rect(b[1], page_rect) for b in eval_term_boxes])

                # Identify llm_answer boxes
                if llm_output.extracted_text is not None:
                    llm_answer_boxes = [
                        [i[0], i[1]]
                        for i in text_boundingbox
                        if any(
                            ext_text[0].split("\n")[-1] in i[0]
                            for ext_text in llm_output.extracted_text
                        )
                    ]
                    llm_answer_rects.extend([get_normalized_rect(b[1], page_rect) for b in llm_answer_boxes])

            # Merge rectangles to avoid overlapping highlights
            district_rects = merge_rects(district_rects)
            eval_term_rects = merge_rects(eval_term_rects)
            llm_answer_rects = merge_rects(llm_answer_rects)

            # Extend rectangles
            extended_district_rects = [
                i for rect in district_rects for i in extend_rect(rect, page_rect)
            ]
            extended_eval_term_rects = [
                i for rect in eval_term_rects for i in extend_rect(rect, page_rect)
            ]

            # Determine overlaps
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
                        overlapping_district_rects = [
                            rect
                            for rect in district_rects
                            if any(llm_rect.intersects(i) for i in extend_rect(rect, page_rect))
                        ]
                        overlapping_eval_term_rects = [
                            rect
                            for rect in eval_term_rects
                            if any(llm_rect.intersects(i) for i in extend_rect(rect, page_rect))
                        ]

                        for rect in overlapping_district_rects:
                            to_be_highlighted_district_rects.append([rect, 0.2])
                        for rect in overlapping_eval_term_rects:
                            to_be_highlighted_eval_term_rects.append([rect, 0.2])

                        to_be_highlighted_llm_answer_rects.append([llm_rect, 0.5])
            else:
                to_be_highlighted_district_rects = [
                    [rect, 0.2] for rect in district_rects
                ]
                to_be_highlighted_eval_term_rects = [
                    [rect, 0.2] for rect in eval_term_rects
                ]
                to_be_highlighted_llm_answer_rects = [
                    [rect, 0.2] for rect in llm_answer_rects
                ]

            # Apply highlights
            district_color = (1, 0, 0)  # Red
            eval_term_color = (0, 0, 1)  # Blue
            llm_answer_color = (0, 1, 0)  # Green

            for rect, opacity in to_be_highlighted_district_rects:
                page.draw_rect(
                    rect,
                    color=district_color,
                    fill=None,
                    fill_opacity=0,
                    width=4,
                    stroke_opacity=opacity,
                )

            for rect, opacity in to_be_highlighted_eval_term_rects:
                page.draw_rect(
                    rect,
                    color=eval_term_color,
                    fill=None,
                    fill_opacity=0,
                    width=4,
                    stroke_opacity=opacity,
                )

            for rect, opacity in to_be_highlighted_llm_answer_rects:
                page.draw_rect(
                    rect,
                    color=llm_answer_color,
                    fill=None,
                    fill_opacity=0,
                    width=4,
                    stroke_opacity=opacity,
                )

        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        # Render the page to a PIL Image
        pix = page.get_pixmap(matrix=mat, clip=page_rect)
        img_bytes = pix.pil_tobytes(format="PNG")
        edited_pages.append(img_bytes)
    return edited_pages


pdf_file = f"https://zoning-nan.s3.us-east-2.amazonaws.com/pdf/north_carolina/{town_name}-zoning-code.pdf"
ocr_file_url = f"https://zoning-nan.s3.us-east-2.amazonaws.com/ocr/north_carolina/{place.town}.json"

if "doc" not in st.session_state or st.session_state["doc"] is None:
    with st.spinner("Downloading PDF for new town..."):
        file_content = download_file_with_progress(pdf_file)
    st.session_state["doc"] = fitz.open(stream=file_content, filetype="pdf")

if "ocr_info" not in st.session_state or not st.session_state["ocr_info"]:
    ocr_file_url = f"https://zoning-nan.s3.us-east-2.amazonaws.com/ocr/north_carolina/{town_name}.json"
    with st.spinner("Downloading OCR info for new town..."):
        file_content = download_file_with_progress(ocr_file_url)
    st.session_state["ocr_info"] = json.loads(file_content)

if "format_ocr_result" not in st.session_state or st.session_state["format_ocr_result"] is None:
    with st.spinner("Downloading Format OCR info for new town..."):
        file_content = download_file_with_progress(f"{s3_prefix}/format_ocr/{town_name}.json")
    st.session_state["format_ocr_result"] = FormatOCR.model_construct(**json.loads(file_content))

extract_blocks = [b for d in st.session_state["ocr_info"] for b in d["Blocks"]]
to_be_highlighted_pages = get_edited_pages(all_showed_pages, all_highlight_info, st.session_state["format_ocr_result"], extract_blocks,
                                           llm_output, place, eval_term, selected_state)

page_img_cols = st.columns(3)
for k in range(len(to_be_highlighted_pages) // 3 + 1):
    for j in range(3):
        i = k * 3 + j
        if i >= len(to_be_highlighted_pages):
            continue
        page_img_cols[j].image(
            to_be_highlighted_pages[i],
            use_column_width=True,
        )

st.divider()


# write data
def write_data(human_feedback: str) -> bool:
    batch = st.session_state["current_batch"]

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

    for item in batch:
        town_name = item['town_name']
        eval_term = item['eval_term']
        place = Place.from_str(item['district'])

        current_viewing_data_name = f"{item['eval_term']}__{item['district'].replace(' ', '+')}.json"
        visualized_data = {
            k: [
                X.model_construct(**json.loads(requests.get(f"{s3_prefix}/{k}/{current_viewing_data_name}").text))
            ]
            for k, X in [
                ("search", SearchResult),
                ("prompt", PromptResult),
                ("llm", LLMInferenceResult),
                ("normalization", NormalizedLLMInferenceResult),
                ("eval", EvalResult),
            ]
        }

        # loading info
        normalized_llm_inference_result = visualized_data["normalization"][0]
        normalized_llm_output = normalized_llm_inference_result.normalized_llm_outputs[0]
        norm = normalized_llm_output.llm_output.answer

        district_full_name = place.district_full_name
        district_short_name = place.district_short_name

        d = {
            "analyst_name": st.session_state["analyst_name"],
            "state": selected_state,
            "town": town_name,
            "district_full_name": district_full_name,
            "district_short_name": district_short_name,
            "eval_term": format_eval_term[eval_term],
            "llm_answer": norm,
            "human_feedback": human_feedback,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": elapsed_sec,
        }

        try:
            doc_ref = db.collection(selected_state)
            _, result = doc_ref.add(d)
            result.get()  # Wait for acknowledgment
        except GoogleAPIError as e:
            st.error(f"Error writing to Firestore: {e}")
            return False
        except FirebaseError as e:
            st.error(f"Firebase SDK Error: {e}")
            return False

    st.session_state["start_time"] = time.time()  # Reset the timer
    st.toast("Going to next data in 2 seconds", icon="🚀")
    st.toast("Data successfully written to database!", icon="🎉")
    return True


# Function to jump to the next batch
def jump_to_next_batch():
    # Update the labelled data
    labelled_data = get_firebase_data(
        selected_state, {"analyst_name": ["==", st.session_state["analyst_name"]]}
    )
    total_num_batches, finished_num_batches, next_batch_info = get_next_unlabeled_batch(
        labelled_data, all_batches
    )
    if next_batch_info:
        idx, batch = next_batch_info
        st.session_state["current_batch"] = batch
        next_town_name = batch[0]['town_name']
        if st.session_state["current_town"] != next_town_name:
            st.session_state["pdf_data"] = None
            st.session_state["ocr_info"] = None  # Reset the OCR info
            st.session_state["finish-town-opened"] = True
            st.session_state["model_next_town_text"] = (
                f"🎉 You've finished the data for {format_town(st.session_state['current_town'])}!\n"
                f"Next town: {format_town(next_town_name)}."
            )
            st.session_state["current_town"] = next_town_name
    else:
        st.subheader("🎉 You've reached the end of the data!")
        st.download_button(
            label="Download all labeled data (CSV)",
            data=prepare_data_for_download(
                selected_state,
                filters={"analyst_name": ["==", st.session_state["analyst_name"]]},
            ).to_csv(index=False),
            file_name=f"{selected_state}_data.csv",
            mime="text/csv",
        )
        st.stop()


# Modal to notify about finishing a town
model_next_town = Modal("", key="finish-town", padding=20, max_width=744)
if "finish-town-opened" not in st.session_state:
    st.session_state["finish-town-opened"] = False
if st.session_state["finish-town-opened"]:
    with model_next_town.container():
        st.header(st.session_state["model_next_town_text"])
        st.session_state["finish-town-opened"] = False  # Reset the flag

# Buttons for labeling
with st.container():
    correct_col, not_sure_col, wrong_col = st.columns(3)


    def button_callback(feedback):
        def _button_callback():
            if write_data(feedback):
                jump_to_next_batch()

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

# Display the next batch preview
# Update the labelled data
labelled_data = get_firebase_data(
    selected_state, {"analyst_name": ["==", st.session_state["analyst_name"]]}
)

# Display the next batch preview
next_batch_index = st.session_state["current_batch_index"] + 1

if next_batch_index < len(all_batches):
    next_batch = all_batches[next_batch_index]
    next_item = next_batch[0]
    next_place = Place.from_str(next_item['district'])
    st.html(
        f"""
        <h2 style="text-align: center; font-size: 2.5em;">
            <em>Next item starts with: {format_eval_term[next_item['eval_term']]}</em> for the
            <em>{next_place.district_full_name} ({next_place.district_short_name})</em>
            District in <em>{format_town(next_item['town_name'])}</em>
        </h2>
    """
    )
else:
    st.write("No more items to label")

# Link to PDF file
pdf_file = f"https://zoning-nan.s3.us-east-2.amazonaws.com/pdf/north_carolina/{town_name}-zoning-code.pdf"
st.link_button(f"PDF Link for {format_town(town_name)}", pdf_file)

if finished_num_batches > 0:
    st.download_button(
        label="Download all labeled data (CSV)",
        data=prepare_data_for_download(
            selected_state,
            filters={"analyst_name": ["==", st.session_state["analyst_name"]]},
        ).to_csv(index=False),
        file_name=f"{selected_state}_data.csv",
        mime="text/csv",
    )

