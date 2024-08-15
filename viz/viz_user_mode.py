import glob
import json
import pandas as pd
import fitz  # PyMuPDF
import requests
import streamlit as st

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

from google.cloud import firestore

db = firestore.Client.from_service_account_info(st.secrets["firebase"]['my_project_settings'])


state_experiment_map = {
    "Connecticut": "results/textract_es_gpt4_connecticut_search_range_3",
    "Texas": "results/textract_es_gpt4_texas_search_range_3",
    "North Carolina": "results/textract_es_gpt4_north_carolina_search_range_3",
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


def write_data(human_feedback: str):
    if not analyst_name or analyst_name == "":
        st.toast("Please enter your name", icon="🚨")
    else:
        town_name, district_short_name, district_full_name = place.split("__")
        d = {
            "analyst_name": analyst_name,
            "state": selected_state,
            "town": town_name,
            "district_full_name": district_full_name,
            "district_short_name": district_short_name,
            "eval_term": eval_term,
            "human_feedback": human_feedback,
            "date": pd.Timestamp.now(),
        }
        
        doc_ref = db.collection(selected_state)
        
        doc_ref.add(d)
        
        st.toast("Data successfully written to database!", icon='🎉')

def get_firebase_csv_data(state: str):
    doc_ref = db.collection(selected_state)
    
    docs = doc_ref.get()
    
    data = []
    
    # Iterate through documents and extract data
    for doc in docs:
        data.append(doc.to_dict())
    
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    
    return df.to_csv(index=False)
    

# Sidebar config
with st.sidebar:
    # Step 0: enter your name
    analyst_name = st.text_input("Please Enter your name", placeholder="")
    
    # Step 1: load files

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

    s3_pdf_dir = f"https://zoning-nan.s3.us-east-2.amazonaws.com/pdf/{format_state(selected_state)}"

    experiment_dir = state_experiment_map[selected_state]

    all_results = {
        k: [
            X.model_construct(**json.load(open(i)))
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

    all_eval_terms = list(set([i.eval_term for i in all_results["eval"]]))
    all_places = list(set(str(i.place) for i in all_results["eval"]))

    def show_fullname_shortname(place):
        town, district_short_name, district_full_name = place.split("__")
        return f"{"-".join([i[0].upper()+i[1:] for i in town.split("-")])}, {district_full_name} ({district_short_name})"

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

    st.write(
        "Number of :blue-background[all] selected data: ",
        len(all_places),
    )

    # Step 2: Config
    st.divider()
    st.subheader("Step 2: Config", divider="rainbow")

    eval_term = st.radio(
        "Choosing :blue-background[Eval Term] you want to check",
        [format_eval_term[i] for i in all_eval_terms],
        key="eval_term",
        index=0,
    )
    eval_term = inverse_format_eval_term[eval_term]

    # eval_type = st.radio(
    #     "Choosing :orange-background[Data Result Type] you want to check",
    #     (
    #         "all",
    #         "correct",
    #         "only_wrong_answer",
    #         "only_wrong_page",
    #         "wrong_answer_and_page",
    #     ),
    #     key="eval_type",
    #     index=0,
    # )

    selected_data = [i for _, i in sorted(all_data_by_eval_term[eval_term].items())]

    # if eval_type == "correct":
    #     selected_data = [
    #         i
    #         for i in selected_data
    #         if i["eval"][0].answer_correct and i["eval"][0].page_in_range
    #     ]
    # if eval_type == "only_wrong_answer":
    #     selected_data = [
    #         i
    #         for i in selected_data
    #         if not i["eval"][0].answer_correct and i["eval"][0].page_in_range
    #     ]
    # if eval_type == "only_wrong_page":
    #     selected_data = [
    #         i
    #         for i in selected_data
    #         if i["eval"][0].answer_correct and not i["eval"][0].page_in_range
    #     ]
    # if eval_type == "wrong_answer_and_page":
    #     selected_data = [
    #         i
    #         for i in selected_data
    #         if not i["eval"][0].answer_correct and not i["eval"][0].page_in_range
    #     ]

    # Step 3: Select one data to check
    st.divider()
    st.subheader("Step 3: Select one data to check", divider="rainbow")

    place = st.radio(
        "Which data to check?",
        (format_place_map[term["place"]] for term in selected_data),
    )
    
    st.subheader("Step 4: Download all labeled data", divider="rainbow")
    st.download_button(
        label="Download CSV",
        data=get_firebase_csv_data(selected_state),
        file_name=f"{selected_state}_data.csv",
        mime="text/csv"
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

ground_truth = eval_result.ground_truth
ground_truth_orig = eval_result.ground_truth_orig
ground_truth_page = eval_result.ground_truth_page
answer_correct = eval_result.answer_correct
page_in_range = eval_result.page_in_range


pdf_file = target_pdf(place.town, s3_pdf_dir)

r = requests.get(pdf_file)
doc = fitz.open(stream=r.content, filetype="pdf")

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

# # show
# st.subheader(f"Town: {place.town}")
# st.subheader(f"District: {place.district_full_name}")
# st.subheader(f"District Abbreviation: {place.district_short_name}")
# st.subheader(f"Eval Term: {eval_term}")
# st.divider()


# summary_col, search_col = st.columns(2)
# with summary_col:

# st.write("Zoning AI Suggests Search Pages:(Red is page with answer)")
# cols = st.columns(len(jump_pages))
# for i in range(len(jump_pages)):
#     page_num = jump_pages[i]
#     if page_num in highlight_text_pages:
#         if cols[i].button(str(page_num), args=(f"{page_num}",), type="primary"):
#             current_page = page_num
#     else:
#         if cols[i].button(
#             str(page_num),
#             args=(f"{page_num}",),
#         ):
#             current_page = page_num
st.write(
    ":blue-background[Town]: {},:blue-background[District]: {} ({})".format(
        "-".join([i[0].upper() + i[1:] for i in place.town.split("-")]),
        place.district_full_name,
        place.district_short_name,
    )
)
st.write(":blue-background[Eval Term]: {}".format(format_eval_term[eval_term]))

st.write(
    ":blue-background[LLM Answer]: {}".format(
        normalized_llm_output.normalized_answer
    )
)
st.write(":blue-background[LLM Rationale]: {}".format(llm_output.rationale))

st.link_button("PDF Link", pdf_file)
# current_page = st.number_input(
#     "Selected page",
#     min_value=1,
#     max_value=len(doc),
#     value=current_page,
# )

# current_page = highlight_text_pages[0]

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
    format_ocr_result = FormatOCR.model_construct(**json.load(open(format_ocr_file)))

    # load ocr 
    try:
        ocr_file_url = f"https://zoning-nan.s3.us-east-2.amazonaws.com/ocr/{format_state(selected_state)}/{place.town}.json"
        # glob.glob(f"data/{format_state(selected_state)}/ocr/{place.town}.json")
        response = requests.get(ocr_file_url)
        response.raise_for_status()
        ocr_info = response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        ocr_info = []

    extract_blocks = [b for d in ocr_info for b in d["Blocks"]]

    edited_pages = []
    for show_page in showed_pages:
        page = doc.load_page(show_page - 1)
        page_rect = page.rect
        # for zoom in
        page_rect += (50, 50, -50, -50)
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
                            (b["Left"] + b["Width"] + 50) * page_rect.width,
                            (b["Top"] + b["Height"] + 50) * page_rect.height,
                        )
                    else:
                        raise ValueError("State not supported")
                    page.draw_rect(normalized_rect, fill=color, width=1, stroke_opacity=0, fill_opacity=0.15)

        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        
        # Render the page to a PIL Image
        pix = page.get_pixmap(matrix=mat, clip=page_rect)
        # pix = page.get_pixmap()
        img_bytes = pix.pil_tobytes(format="PNG")
        edited_pages.append(img_bytes)

    page_img_cols = st.columns(len(showed_pages))

    for i in range(len(showed_pages)):
        page_img_cols[i].image(
            edited_pages[i],
            # caption=f"Page {showed_pages[i]}",
            use_column_width=True,
            # width=400
        )
    # pdf_viewer = st.empty()
    # pdf_viewer.image(
    #     img_bytes,
    #     caption=f"Page {current_page}",
    #     use_column_width=True,
    #     # width=400
    # )


    # # with search_col:
    # st.write("District is highlighted in :red-background[red]")
    # st.write("Eval Term is highlighted in :blue-background[blue]")
    # st.write("LLM answer is highlighted in :green-background[green]")

    # with search_col:

st.divider()
with st.container(border=True):
    # st.subheader("Current data")
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

# st.divider()

# st.title("More Details")
# with st.container(height=700):
#     st.subheader("LLM Inference Results")

#     llm_answer_col, llm_response_detail_col = st.columns(2)
#     with llm_answer_col:
#         st.write("*Answer*")
#     with llm_response_detail_col:
#         st.write("*Full Details*")

#     with st.container(height=700, border=False):
#         llm_answer_col, llm_response_detail_col = st.columns(2)
#         with llm_answer_col:
#             st.write(
#                 "Search Page: :orange-background[{}]".format(
#                     entire_search_page_range
#                 )
#             )
#             st.write(
#                 "Highlighted Pages: :orange-background[{}]".format(
#                     highlight_text_pages
#                 )
#             )
#             st.write(
#                 "Normalized LLM Answer: :orange-background[{}]".format(
#                     normalized_llm_output.normalized_answer
#                 )
#             )
#             st.json(
#                 {
#                     "extracted_text": llm_output.extracted_text,
#                     "rationale": llm_output.rationale,
#                     "answer": llm_output.answer,
#                     "normalized_answer": normalized_llm_output.normalized_answer,
#                 }
#             )
#         with llm_response_detail_col:
#             st.json(
#                 {
#                     "input_prompt": [
#                         input_prompt.system_prompt,
#                         input_prompt.user_prompt,
#                     ],
#                     "raw_model_response": llm_output.raw_model_response,
#                 },
#                 expanded=False,
#             )
#         st.divider()

#     st.subheader("Search Results before merge")
#     search_meta_col, search_text_col = st.columns(2)
#     with search_meta_col:
#         st.write("*Relevance Score & Page Range*")
#     with search_text_col:
#         st.write("*Text & Highlight*")
#     with st.container(height=400, border=False):
#         for idx, search_result in enumerate(search_result.search_matches):
#             search_meta_col, search_text_col = st.columns(2)
#             with search_meta_col:
#                 st.write(
#                     f"Relevance Score :orange-background[{search_result.score}]"
#                 )
#                 st.write(
#                     f"Page Range :orange-background[{sorted(search_result.page_range)}]"
#                 )
#             with search_text_col:
#                 st.json(
#                     {
#                         "text": search_result.text,
#                         "highlight": search_result.highlight,
#                     },
#                     expanded=False,
#                 )
#             st.divider()
