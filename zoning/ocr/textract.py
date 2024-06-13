import json
import os
import time
from typing import Generator, List, Tuple

import boto3
import tqdm
from omegaconf import DictConfig
from tqdm.contrib.concurrent import process_map, thread_map

from zoning.class_types import (
    ExtractionEntities,
    ExtractionEntity,
    ExtractionResult,
    ExtractionResults,
)
from zoning.ocr.base_extractor import Extractor


class TextractExtractor(Extractor):
    def __init__(self, extractor_config: DictConfig):
        super().__init__(extractor_config)
        if self.config.extract.run_ocr:
            self.extractor = boto3.client("textract")

    def start_job(self, s3_bucket_name: str) -> str:
        """Runs Textract's StartDocumentAnalysis action and specifies an s3
        bucket to dump output."""
        response = self.extractor.start_document_analysis(
            DocumentLocation={
                "S3Object": {
                    "Bucket": self.config.extract.input_document_s3_bucket,
                    "Name": s3_bucket_name,
                }
            },
            FeatureTypes=self.config.extract.feature_types,
        )

        return response["JobId"]

    def get_job_status(self, job_id: str) -> Generator[Tuple[str, str], None, None]:
        """' Checks whether document analysis still in progress."""
        status: str = "IN_PROGRESS"
        while status == "IN_PROGRESS":
            time.sleep(5)
            response = self.extractor.get_document_analysis(JobId=job_id)
            status = response["JobStatus"]
            yield status, response.get("StatusMessage", None)

    def get_job_results(self, job_id: str) -> Generator[dict, None, None]:
        """If document analysis complete, runs Textract's GetDocumentAnalysis
        action and pulls JSON results to be stored in s3 bucket designated
        above."""
        response = self.extractor.get_document_analysis(JobId=job_id)
        nextToken = response.get("NextToken", None)
        yield response

        while nextToken is not None:
            response = self.extractor.get_document_analysis(
                JobId=job_id, NextToken=nextToken
            )
            nextToken = response.get("NextToken", None)
            yield response

    def collect_relations(self, w) -> List[str]:
        rels = w["Relationships"] if "Relationships" in w else []
        ids = []
        for r in rels if rels else []:
            for id in r["Ids"]:
                ids.append(id)
        return ids

    def _extract(self, target: ExtractionEntity) -> None:
        if self.config.extract.pdf_name_prefix_in_s3_bucket:
            s3_bucket_name = (
                self.config.extract.pdf_name_prefix_in_s3_bucket + target.pdf_file
            )
        else:
            s3_bucket_name = target.pdf_file
        job_id = self.start_job(s3_bucket_name)
        for s in self.get_job_status(job_id):
            status, status_message = s
            if status == "FAILED":
                print(
                    f"Job {job_id} on file {target.pdf_file} FAILED. Reason: {status_message}"
                )
            elif status == "SUCCEEDED":
                result = list(self.get_job_results(job_id))
                with open(target.ocr_result_file, "w", encoding="utf-8") as f:
                    json.dump(result, f)
                print(
                    f"Job {job_id} on file {target.pdf_file} SUCCEEDED. Write to {target.ocr_result_file}"
                )

    def process_ocr_results(self, target: ExtractionEntity) -> None:
        """
        Inputs:
            target (ExtractionEntity): TODO
        Returns: TODO
        """
        with open(target.ocr_result_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding {target.ocr_result_file}")
                return None

        extract_blocks = [b for d in data for b in d["Blocks"]]

        entities = ExtractionResults(ents=[], seen=set(), relations={})
        rows = []
        for w in tqdm.tqdm(extract_blocks):
            if w["BlockType"] in ["LINE", "WORD", "CELL", "MERGED_CELL"]:
                e = ExtractionResult(
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
                    rows.append(
                        {
                            self.config.index.index_key: f"{target.name}",
                            "Page": w["Page"] - 1,
                            "Text": str(entities),
                        }
                    )
                entities = ExtractionResults(ents=[], seen=set(), relations={})
            elif w["BlockType"] == "TABLE":
                pass
            else:
                continue

        if len(entities.ents) > 0:
            rows.append(
                {
                    self.config.index.index_key: f"{target.name}",
                    "Page": w["Page"],
                    "Text": str(entities),
                }
            )

        with open(target.dataset_file, "w") as f:
            json.dump(rows, f)

    def extract(self, extract_targets: ExtractionEntities) -> None:
        if self.config.extract.run_ocr:
            thread_map(self._extract, extract_targets)
        assert (
            len(os.listdir(extract_targets.ocr_result_dir)) > 0
        ), "No OCR results found"

        # OCR result from textract is a list of json objects containing unneeded information, we need to extract the text from it
        process_map(self.process_ocr_results, extract_targets.targets)
