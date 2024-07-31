from typing import Any, Dict, List

import hydra
import tqdm
from omegaconf import OmegaConf

from zoning.class_types import FormatOCR, OCRBlock, OCRPage, ZoningConfig
from zoning.utils import process


def collect_relations(w) -> List[str]:
    rels = w["Relationships"] if "Relationships" in w else []
    ids = []
    for r in rels if rels else []:
        for id in r["Ids"]:
            ids.append(id)
    return ids


OCROutput = List[Dict[str, Any]]


def process_ocr_result(data: OCROutput, town_name: str) -> FormatOCR:
    extract_blocks = [b for d in data for b in d["Blocks"]]

    ocr_page = OCRPage()
    formatted_ocr = FormatOCR(pages=[], town_name=town_name)
    for w in tqdm.tqdm(extract_blocks):
        if w["BlockType"] in ["LINE", "WORD", "CELL", "MERGED_CELL"]:
            ocr_block = OCRBlock(
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
            ocr_page.add(ocr_block)
        elif w["BlockType"] == "PAGE":
            if len(ocr_page.ents) > 0:

                # since the key name is not unique, we are unable to use a BaseModel for it
                # so here, we did not use a type hint for the key name
                ocr_page.page = w["Page"] - 1
                formatted_ocr.pages.append(ocr_page.make_dict())
            ocr_page = OCRPage()
        elif w["BlockType"] == "TABLE":
            pass
        else:
            continue

    if len(ocr_page.ents) > 0:
        # since the key name is not unique, we are unable to use a BaseModel for it
        # so here, we did not use a type hint for the key name
        ocr_page.page = w["Page"]
        formatted_ocr.pages.append(ocr_page.make_dict())
    return formatted_ocr


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: ZoningConfig):
    """Format ocr results into structure data format.

    Configs:
        - global_config: GlobalConfig.
        - format_ocr_config: FormatOCRConfig

    Input File Format:
        Raw OCR from Textract
        config.ocr_dir

    Output File Format:
        FormatOCR
        config.format_ocr_dir
    """

    # Parse the config
    config = OmegaConf.to_object(config)
    global_config = ZoningConfig(config=config).global_config
    # format_ocr_config = ZoningConfig(config=config).format_ocr_config

    # Construct the input data
    process(
        global_config.target_town_file,
        global_config.ocr_dir,
        global_config.format_ocr_dir,
        process_ocr_result,
    )


if __name__ == "__main__":
    main()
