"""
@Time ： 9/1/2024 12:49 AM
@Auth ： Yizhi Hao
@File ：zoning_routes
@IDE ：PyCharm
"""

from flask import Blueprint, request, jsonify
from viz_v2.backend.models.firestore_client import FirestoreClient
from viz_v2.backend.models.data_models import ZoningData
from viz_v2.backend.utils import load_evals_data
import datetime
import time

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
