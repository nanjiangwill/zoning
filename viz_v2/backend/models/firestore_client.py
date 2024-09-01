"""
@Time ： 9/1/2024 12:46 AM
@Auth ： Yizhi Hao
@File ：firestore
@IDE ：PyCharm
"""

from google.cloud import firestore
from viz_v2.backend.config import Config

class FirestoreClient:
    def __init__(self):
        self.client = firestore.Client.from_service_account_json(Config.FIREBASE_CREDENTIALS_PATH)

    def get_collection(self, collection_name):
        return self.client.collection(collection_name)
