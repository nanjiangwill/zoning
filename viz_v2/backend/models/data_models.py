"""
@Time ： 9/1/2024 12:47 AM
@Auth ： Yizhi Hao
@File ：data_models
@IDE ：PyCharm
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ZoningData:
    analyst_name: str
    state: str
    town: str
    district_full_name: str
    district_short_name: str
    eval_term: str
    human_feedback: str
    date: str
    elapsed_sec: Optional[float] = None
