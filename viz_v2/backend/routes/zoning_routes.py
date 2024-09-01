"""
@Time ： 9/1/2024 12:49 AM
@Auth ： Yizhi Hao
@File ：zoning_routes
@IDE ：PyCharm
"""

from flask import Blueprint, request, jsonify
from viz_v2.backend.models.firestore_client import FirestoreClient
from viz_v2.backend.models.data_models import ZoningData
import datetime
import time

zoning_bp = Blueprint('zoning', __name__)

db_client = FirestoreClient()

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

    doc_ref = db_client.get_collection(zoning_data.state).add(zoning_data.__dict__)

    return jsonify({"message": "Data successfully written to database!"}), 201

@zoning_bp.route('/fetch', methods=['GET'])
def fetch_data():
    state = request.args.get('state')
    if not state:
        return jsonify({"error": "State is required"}), 400

    docs = db_client.get_collection(state).get()
    data = [doc.to_dict() for doc in docs]

    return jsonify(data), 200
