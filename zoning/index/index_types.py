import json
import os
from dataclasses import dataclass
from typing import Dict, Generator

from omegaconf import DictConfig


@dataclass
class ElasticSearchIndexData:
    index: str
    id: str
    document: Dict[str, str]
    request_timeout: int = 30


@dataclass
class IndexEntity:
    name: str
    dataset_dir: str
    index_range: int

    def __init__(self, name, dataset_dir, index_range):
        self.name = name
        self.dataset_dir = dataset_dir
        self.index_range = index_range
        self.dataset_file = os.path.join(dataset_dir, f"{name}.json")
        assert os.path.exists(
            self.dataset_file
        ), f"Dataset file {self.dataset_file} does not exist"

    def get_index_data(self) -> Generator[ElasticSearchIndexData, None, None]:
        with open(self.dataset_file, "r") as f:
            index_data = json.load(f)
        for idx in range(len(index_data)):
            text = ""
            for j in range(self.index_range):
                if idx + j >= len(index_data):
                    break
                text += f"\nNEW PAGE {idx + j + 1}\n" + index_data[idx + j]["Text"]
            yield ElasticSearchIndexData(
                index=self.name,
                id=idx + 1,
                document={"Page": idx + 1, "Text": text},
                request_timeout=30,
            )


@dataclass
class IndexEntities:
    index_entities: list[IndexEntity]
    dataset_dir: str

    def __init__(self, config: DictConfig):
        self.dataset_dir = config.dataset_dir
        self.index_range = (
            config.index.index_range if "index_range" in config.index else 1
        )
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset directory {self.dataset_dir} does not exist"
        assert os.listdir(
            self.dataset_dir
        ), f"Dataset directory {self.dataset_dir} is empty"
        self.index_entities = self.load_entities(
            config.target_names_file
        )  # we index every extracted file

    def load_entities(self, target_names_file: str) -> list[IndexEntity]:
        with open(target_names_file, "r") as f:
            target_names = json.load(f)

        # simple detect if there are any files that are not extracted
        # can be deleted later
        missing_files = [
            name
            for name in target_names
            if not os.path.exists(os.path.join(self.dataset_dir, f"{name}.json"))
        ]
        print("Missing files: ", missing_files)
        print("Total missing files: ", len(missing_files))
        print("Total files: ", len(target_names))

        return [
            IndexEntity(name, self.dataset_dir, self.index_range)
            for name in target_names
            if os.path.exists(os.path.join(self.dataset_dir, f"{name}.json"))
        ]
