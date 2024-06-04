from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, cast


@dataclass
class Extractor:
    def __init__(self, extractor_config):
        self.config = extractor_config
        self.name = extractor_config.extract.name

    def extract(self, state_all_towns_names):
        pass

    def post_extract(self, state_all_towns_names):
        pass


@dataclass
class Entity:
    id: str
    text: str
    typ: str
    relationships: List[str]
    position: Tuple[int, int]


@dataclass
class Entities:
    ents: List[Entity]
    seen: Set[str]
    relations: Dict[str, List[Entity]]

    def add(self, entity: Entity):
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
