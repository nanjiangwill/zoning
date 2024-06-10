import json
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from dataclasses_json import dataclass_json
from omegaconf import DictConfig


@dataclass
class ExtractionEntity:
    name: str
    pdf_file: str
    ocr_result_file: str
    dataset_file: str

    def __init__(self, name, pdf_dir, ocr_result_dir, dataset_dir):
        self.name = name
        self.pdf_file = os.path.join(pdf_dir, f"{name}.pdf")
        self.ocr_result_file = os.path.join(ocr_result_dir, f"{name}.json")
        self.dataset_file = os.path.join(dataset_dir, f"{name}.json")


@dataclass
class ExtractionEntities:
    targets: list[ExtractionEntity]
    pdf_dir: str
    ocr_result_dir: str
    dataset_dir: str

    def __init__(self, config: DictConfig):
        self.pdf_dir = config.pdf_dir
        self.ocr_result_dir = config.ocr_result_dir
        self.dataset_dir = config.dataset_dir
        self.targets = self.load_targets(config.target_names_file)
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.ocr_result_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)

    def load_targets(self, target_names_file: str) -> list[ExtractionEntity]:
        with open(target_names_file, "r") as f:
            target_names = json.load(f)
        return [
            ExtractionEntity(name, self.pdf_dir, self.ocr_result_dir, self.dataset_dir)
            for name in target_names
        ]

    def get_all_names(self) -> list[str]:
        return [target.name for target in self.targets]

    def get_all_pdf_files(self) -> list[str]:
        return [target.pdf_file for target in self.targets]

    def get_all_ocr_result_files(self) -> list[str]:
        return [target.ocr_result_file for target in self.targets]

    def get_all_dataset_files(self) -> list[str]:
        return [target.dataset_file for target in self.targets]


# TODO
@dataclass_json
@dataclass
class ExtractionResult:
    id: str
    text: str
    typ: str
    relationships: List[str]
    position: Tuple[int, int]


@dataclass_json
@dataclass
class ExtractionResults:
    ents: List[ExtractionResult]
    seen: Set[str]
    relations: Dict[str, List[ExtractionResult]]

    def add(self, entity: ExtractionResult):
        if entity.id in self.seen:
            return
        self.ents.append(entity)
        self.seen.add(entity.id)
        for r in entity.relationships:
            self.relations.setdefault(r, [])
            self.relations[r].append(entity)

    def __str__(self) -> str:
        out = ""
        for e in self.ents:
            if e.typ == "LINE":
                in_cell = [
                    o
                    for r in e.relationships
                    for o in self.relations[r]
                    if o.typ == "CELL"
                ]
                if not in_cell:
                    out += e.text + "\n"
            if e.typ == "CELL":
                lines = [
                    o
                    for r in e.relationships
                    for o in self.relations[r]
                    if o.typ == "LINE"
                ]

                out += f"CELL {e.position}: \n"
                seen = set()
                for o in lines:
                    if o.id in seen:
                        continue
                    seen.add(o.id)
                    out += o.text + "\n"
        return out
