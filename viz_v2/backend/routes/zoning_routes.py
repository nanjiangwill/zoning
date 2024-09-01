"""
@Time ： 9/1/2024 12:49 AM
@Auth ： Yizhi Hao
@File ：zoning_routes
@IDE ：PyCharm
"""

from flask import Blueprint, request, jsonify, send_file
from viz_v2.backend.models.firestore_client import FirestoreClient
from viz_v2.backend.models.data_models import ZoningData
from viz_v2.backend.utils import load_evals_data
from viz_v2.backend.config import Config
from zoning.class_types import Place, FormatOCR
from zoning.utils import target_pdf, expand_term
import datetime
import time
import fitz
import json
import glob
from io import BytesIO

zoning_bp = Blueprint('zoning', __name__)

db_client = FirestoreClient()

# Preload data at startup
data_store = load_evals_data()

@zoning_bp.route('/write', methods=['POST'])
def write_data():
    data = request.json
    if not data.get('analyst_name'):
        return jsonify({"error": "Analyst name is required"}), 400

    elapsed_sec = time.time() - data.get("start_time", time.time())

    zoning_data = ZoningData(
        analyst_name=data['analyst_name'],
        state=data['state'],
        town=data['town'],
        district_full_name=data['district_full_name'],
        district_short_name=data['district_short_name'],
        eval_term=data['eval_term'],
        human_feedback=data['human_feedback'],
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        elapsed_sec=elapsed_sec
    )

    # doc_ref = db_client.get_collection(zoning_data.state).add(zoning_data.__dict__)
    print(zoning_data.__dict__)

    return jsonify({"message": "Data successfully written to database!"}), 201


@zoning_bp.route('/state', methods=['GET'])
def get_states():
    """Return all available states."""
    states = list(data_store.keys())
    return jsonify(states), 200


@zoning_bp.route('/evals', methods=['GET'])
def get_evals():
    """Return all available eval terms for a given state."""
    state = request.args.get('state')
    if not state or state not in data_store:
        return jsonify({"error": "Invalid or missing state"}), 400

    eval_terms = list(data_store[state]['all_eval_terms'])
    return jsonify(eval_terms), 200


@zoning_bp.route('/evals/<eval_term>', methods=['GET'])
def get_places(eval_term):
    """Return all available places for a given state and eval term."""
    state = request.args.get('state')
    if not state or state not in data_store:
        return jsonify({"error": "Invalid or missing state"}), 400

    if eval_term not in data_store[state]['all_data_by_eval_term']:
        return jsonify({"error": "Invalid eval term"}), 400

    places = list(data_store[state]['all_data_by_eval_term'][eval_term].keys())
    return jsonify(places), 200


@zoning_bp.route('/pdf_metadata', methods=['GET'])
def get_pdf_metadata():
    """Return metadata for PDF, without the PDF file."""
    state = request.args.get('state')
    eval_term = request.args.get('eval_term')
    place_name = request.args.get('place')

    if not state or not eval_term or not place_name:
        return jsonify({"error": "Missing required parameters"}), 400

    if state not in data_store:
        return jsonify({"error": "Invalid state"}), 400

    # Get the required data
    state_data = data_store[state]
    if eval_term not in state_data['all_data_by_eval_term']:
        return jsonify({"error": "Invalid eval term"}), 400

    place = Place.from_str(place_name)
    visualized_data = state_data['all_data_by_eval_term'][eval_term][place_name]

    # Prepare the data for the response
    format_eval_term = {
        "floor_to_area_ratio": "Floor to Area Ratio",
        "max_height": "Max Height",
        "max_lot_coverage": "Max Lot Coverage",
        "max_lot_coverage_pavement": "Max Lot Coverage Pavement",
        "min_lot_size": "Min Lot Size",
        "min_parking_spaces": "Min Parking Spaces",
        "min_unit_size": "Min Unit Size",
    }

    eval_term_display = format_eval_term[eval_term]
    norm = visualized_data['normalization'][0].normalized_llm_outputs[0].normalized_answer
    norm = norm[0] if isinstance(norm, list) else norm
    llm_output = visualized_data['llm'][0].llm_outputs[0]

    response_data = {
        "eval_term": eval_term_display,
        "district_full_name": place.district_full_name,
        "district_short_name": place.district_short_name,
        "norm": norm,
        "rationale": llm_output.rationale,
    }

    return jsonify(response_data), 200


@zoning_bp.route('/pdf_file', methods=['GET'])
def get_pdf_file():
    """Return the highlighted PDF file, only sending specific pages."""
    state = request.args.get('state')
    eval_term = request.args.get('eval_term')
    place_name = request.args.get('place')

    if not state or not eval_term or not place_name:
        return jsonify({"error": "Missing required parameters"}), 400

    if state not in data_store:
        return jsonify({"error": "Invalid state"}), 400

    # Fetching the necessary data
    state_data = data_store[state]
    if eval_term not in state_data['all_data_by_eval_term']:
        return jsonify({"error": "Invalid eval term"}), 400

    place = Place.from_str(place_name)
    visualized_data = state_data['all_data_by_eval_term'][eval_term][place_name]

    # Extract LLM output and other relevant information
    llm_output = visualized_data['llm'][0].llm_outputs[0]
    search_result = visualized_data['search'][0]
    entire_search_page_range = search_result.entire_search_page_range

    highlight_text_pages = []
    input_prompt = visualized_data['prompt'][0].input_prompts[0]
    if llm_output.extracted_text:
        for i in llm_output.extracted_text:
            page = int(input_prompt.user_prompt.split(i)[0].split("NEW PAGE ")[-1].split("\n")[0])
            highlight_text_pages.append(page)
        highlight_text_pages = sorted(set(highlight_text_pages))

    def get_showed_pages(pages, interval):
        showed_pages = []
        for page in pages:
            showed_pages.extend(range(page - interval, page + interval + 1))
        return sorted(set(showed_pages))

    showed_pages = get_showed_pages(highlight_text_pages, 1)
    if not showed_pages:
        showed_pages = entire_search_page_range.copy()

    # Load the PDF file
    exp_dir = Config.STATE_EXPERIMENT_MAP[state]
    pdf_file = target_pdf(place.town, exp_dir + "/pdf")
    doc = fitz.open(pdf_file)

    # Extract the relevant pages and add highlights
    output_pdf = fitz.open()  # New empty PDF
    format_ocr_file = glob.glob(f"{exp_dir}/format_ocr/{place.town}.json")[0]
    format_ocr_result = FormatOCR.model_construct(**json.loads(open(format_ocr_file).read()))
    ocr_file = glob.glob(f"{exp_dir}/ocr/{place.town}.json")[0]
    ocr_info = json.loads(open(ocr_file).read())

    extract_blocks = [b for d in ocr_info for b in d["Blocks"]]
    for show_page in showed_pages:
        page = doc.load_page(show_page - 1)
        output_page = output_pdf.new_page(width=page.rect.width, height=page.rect.height)
        output_page.show_pdf_page(output_page.rect, doc, show_page - 1)

        page_rect = page.rect
        page_info = [i for i in format_ocr_result.pages if i["page"] == str(show_page)][0]

        load_ocr = any(term in page_info["text"] for term in expand_term(Config.THESAURUS_PATH, eval_term))
        if any(keyword in page_info["text"].lower() for keyword in [place.town.lower(), place.district_full_name.lower(), place.district_short_name.lower()]):
            load_ocr = True

        if load_ocr:
            page_ocr_info = [w for w in extract_blocks if w["Page"] == show_page]
            text_boundingbox = [(w["Text"], w["Geometry"]["BoundingBox"]) for w in page_ocr_info if "Text" in w]

            district_boxs = [
                [i[0], i[1]] for i in text_boundingbox if place.district_full_name.lower() in i[0].lower() or place.district_short_name in i[0]
            ]
            eval_term_boxs = [
                [i[0], i[1]] for i in text_boundingbox if any(term in i[0] for term in expand_term(Config.THESAURUS_PATH, eval_term))
            ]
            llm_answer_boxs = [
                [i[0], i[1]] for i in text_boundingbox if any(text.split("\n")[-1] in i[0] for text in llm_output.extracted_text)
            ]

            district_color = (1, 0, 0)  # Red
            eval_term_color = (0, 0, 1)  # Blue
            llm_answer_color = (0, 1, 0)  # Green

            box_list = [district_boxs, eval_term_boxs, llm_answer_boxs]
            color_list = [district_color, eval_term_color, llm_answer_color]

            for box, color in zip(box_list, color_list):
                for _, b in box:
                    if state == "texas":
                        normalized_rect = fitz.Rect(
                            b["Left"] * page_rect.width,
                            b["Top"] * page_rect.height,
                            (b["Left"] + b["Width"]) * page_rect.width,
                            (b["Top"] + b["Height"]) * page_rect.height,
                        )
                    elif state == "connecticut":
                        normalized_rect = fitz.Rect(
                            (1 - b["Top"] - b["Height"]) * page_rect.height,
                            b["Left"] * page_rect.width,
                            (1 - b["Top"]) * page_rect.height,
                            (b["Left"] + b["Width"]) * page_rect.width,
                        )
                    elif state == "north_carolina":
                        normalized_rect = fitz.Rect(
                            b["Left"] * page_rect.width,
                            b["Top"] * page_rect.height,
                            (b["Left"] + b["Width"]) * page_rect.width,
                            (b["Top"] + b["Height"]) * page_rect.height,
                        )
                    else:
                        raise ValueError("State not supported")

                    output_page.draw_rect(
                        normalized_rect,
                        fill=color,
                        width=1,
                        stroke_opacity=0,
                        fill_opacity=0.15,
                    )

    # Save the output PDF to a BytesIO object
    pdf_bytes = BytesIO()
    output_pdf.save(pdf_bytes, garbage=4)
    pdf_bytes.seek(0)

    return send_file(
        pdf_bytes,
        as_attachment=True,
        download_name="highlighted_zoning_map.pdf",
        mimetype="application/pdf"
    )
