import json
from typing import List

import tqdm
from elasticsearch import Elasticsearch
from tqdm.contrib.concurrent import thread_map

from zoning.class_types import (
    ElasticSearchIndexData,
    IndexConfig,
    IndexEntities,
    IndexEntity,
    OCREntities,
    OCREntity,
    OCRPageResult,
    OCRPageResults,
)
from zoning.index.base_indexer import Indexer


class KeywordIndexer(Indexer):
    def __init__(self, index_config: IndexConfig):
        super().__init__(index_config)
        self.es_client = Elasticsearch(index_config.es_endpoint)

    def _index(self, index_entity: IndexEntity) -> None:
        all_index_data = []
        page_data = index_entity.page_data

        for idx in range(len(page_data)):
            text = ""
            for j in range(self.index_config.index_range):
                if idx + j >= len(page_data):
                    break
                text += f"\nNEW PAGE {idx + j + 1}\n" + page_data[idx + j]["Text"]
            all_index_data.append(
                ElasticSearchIndexData(
                    index=index_entity.name,
                    id=str(idx + 1),
                    document={"Page": str(idx + 1), "Text": text},
                    request_timeout=30,
                )
            )
        # get index data from one file, which contains multiple pages
        for index_data in all_index_data:
            self.es_client.index(
                index=index_data.index,
                id=index_data.id,
                document=index_data.document,
                request_timeout=index_data.request_timeout,
            )

    def collect_relations(self, w) -> List[str]:
        rels = w["Relationships"] if "Relationships" in w else []
        ids = []
        for r in rels if rels else []:
            for id in r["Ids"]:
                ids.append(id)
        return ids

    def process_ocr_result(self, ocr_entity: OCREntity) -> IndexEntity:
        try:
            data = json.load(open(ocr_entity.ocr_results_file))
        except json.JSONDecodeError:
            print(f"Error decoding {ocr_entity.ocr_results_file}")
            return None

        extract_blocks = [b for d in data for b in d["Blocks"]]

        entities = OCRPageResults(ents=[], seen=set(), relations={})
        rows = []
        for w in tqdm.tqdm(extract_blocks):
            if w["BlockType"] in ["LINE", "WORD", "CELL", "MERGED_CELL"]:
                e = OCRPageResult(
                    id=w["Id"],
                    text=w.get("Text", ""),
                    typ=w["BlockType"],
                    relationships=self.collect_relations(w),
                    position=(
                        (w["RowIndex"], w["ColumnIndex"])
                        if "RowIndex" in w and "ColumnIndex" in w
                        else (-1, -1)
                    ),
                )
                entities.add(e)
            elif w["BlockType"] == "PAGE":
                if len(entities.ents) > 0:

                    # since the key name is not unique, we are unable to use a BaseModel for it
                    # so here, we did not use a type hint for the key name
                    rows.append(
                        {
                            self.index_config.index_key: f"{ocr_entity.name}",
                            "Page": w["Page"] - 1,
                            "Text": str(entities),
                        }
                    )
                entities = OCRPageResults(ents=[], seen=set(), relations={})
            elif w["BlockType"] == "TABLE":
                pass
            else:
                continue

        if len(entities.ents) > 0:

            # since the key name is not unique, we are unable to use a BaseModel for it
            # so here, we did not use a type hint for the key name
            rows.append(
                {
                    self.index_config.index_key: f"{ocr_entity.name}",
                    "Page": w["Page"],
                    "Text": str(entities),
                }
            )

        return IndexEntity(name=ocr_entity.name, page_data=rows)

    def index(self, ocr_entities: OCREntities) -> IndexEntities:
        index_entities = thread_map(self.process_ocr_result, ocr_entities.ocr_entities)
        # removing None values
        index_entities = [i for i in index_entities if i is not None]

        print(f"Indexing {len(index_entities)} entities")
        thread_map(self._index, index_entities)

        return IndexEntities(index_entities=index_entities)
