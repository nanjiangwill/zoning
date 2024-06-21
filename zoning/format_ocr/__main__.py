import json
from typing import Any, Dict, List

import hydra
from omegaconf import OmegaConf
import tqdm

from zoning.class_types import ZoningConfig, Page, OCRBlock, FormattedOCR
from zoning.utils import process

def collect_relations(w) -> List[str]:
    rels = w["Relationships"] if "Relationships" in w else []
    ids = []
    for r in rels if rels else []:
        for id in r["Ids"]:
            ids.append(id)
    return ids


OCROutput = List[Dict[str, Any]]

def process_ocr_result(data: OCROutput, town_name: str) -> FormattedOCR:
    extract_blocks = [b for d in data for b in d["Blocks"]]

    page = Page()
    formatted_ocr = FormattedOCR(pages=[], town_name=town_name)
    for w in tqdm.tqdm(extract_blocks):
        if w["BlockType"] in ["LINE", "WORD", "CELL", "MERGED_CELL"]:
            e = OCRBlock(
                id=w["Id"],
                text=w.get("Text", ""),
                typ=w["BlockType"],
                relationships=collect_relations(w),
                position=(
                    (w["RowIndex"], w["ColumnIndex"])
                    if "RowIndex" in w and "ColumnIndex" in w
                    else (-1, -1)
                ),
            )
            page.add(e)
        elif w["BlockType"] == "PAGE":
            if len(page.ents) > 0:

                # since the key name is not unique, we are unable to use a BaseModel for it
                # so here, we did not use a type hint for the key name
                page.page = w["Page"] - 1
                formatted_ocr.pages.append(page.make_string())
                #     {
                #         self.index_config.index_key: f"{ocr_entity.name}",
                #         "Page": w["Page"] - 1,
                #         "Text": str(entities),
                #     }
                # )
            entities = Page()
        elif w["BlockType"] == "TABLE":
            pass
        else:
            continue
    
    if len(entities.ents) > 0:

        # since the key name is not unique, we are unable to use a BaseModel for it
        # so here, we did not use a type hint for the key name
        page.page = w["Page"]
        formatted_ocr.pages.append(page.make_string())
        #     {
        #         self.index_config.index_key: f"{ocr_entity.name}",
        #         "Page": w["Page"],
        #         "Text": str(entities),
        #     }
        # )
    return formatted_ocr

@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """
    Format ocr results into structure data format.

    Configs:
        - global_config: GlobalConfig.
        - ocr_config: OCRConfig

    OCR Input File Format:
        Raw OCR from Textract
        config.ocr_results_dir

    OCR Output File Format:
        OCREntities
        config.format_ocr_results_dir
    """

    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    ocr_config = ZoningConfig(config=config).format_ocr_config

    # Construct the input data
    process(global_config.target_names_file, global_config.ocr_dir, 
            global_config.format_ocr_dir, process_ocr_result)

if __name__ == "__main__":
    main()
