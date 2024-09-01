"""
@Time ： 9/1/2024 12:43 AM
@Auth ： Yizhi Hao
@File ：config
@IDE ：PyCharm
"""

import os

class Config:
    FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH', '../../gcp_key.json')
    FIREBASE_PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID', 'your-firebase-project-id')
    PDF_DIRECTORY = os.getenv('PDF_DIRECTORY', '../../data')
    STATE_EXPERIMENT_MAP = {
        "connecticut": "../../results/textract_es_gpt4_connecticut_search_range_3",
        "texas": "../../results/textract_es_gpt4_texas_search_range_3",
        "north_carolina": "../../results/textract_es_gpt4_north_carolina_search_range_3_old_thesaurus",
    }
    THESAURUS_PATH = os.getenv('THESAURUS_PATH', '../../data/thesaurus.json')
