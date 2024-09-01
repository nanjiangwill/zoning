"""
@Time ： 9/1/2024 12:43 AM
@Auth ： Yizhi Hao
@File ：config
@IDE ：PyCharm
"""

import os

class Config:
    FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH', 'path/to/serviceAccount.json')
    FIREBASE_PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID', 'your-firebase-project-id')
    PDF_DIRECTORY = os.getenv('PDF_DIRECTORY', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))