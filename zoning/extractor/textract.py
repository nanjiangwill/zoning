import json
import os
import time

import boto3
import jsonlines
import tqdm
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from tqdm.contrib.concurrent import process_map, thread_map

from .base_extractor import Entities, Entity, Extractor


class TextractExtractor(Extractor):
    def __init__(self, extractor_config: DictConfig):
        super().__init__(extractor_config)
        self.extractor = boto3.client("textract")

    def start_job(self, town_pdf_path: str) -> str:
        """Runs Textract's StartDocumentAnalysis action and specifies an s3
        bucket to dump output."""
        response = self.extractor.start_document_analysis(
            DocumentLocation={
                "S3Object": {
                    "Bucket": self.config.extract.input_document_s3_bucket,
                    "Name": town_pdf_path,
                }
            },
            FeatureTypes=self.config.extract.feature_types,
        )

        return response["JobId"]

    def get_job_status(self, job_id: str):
        """' Checks whether document analysis still in progress."""
        status: str = "IN_PROGRESS"
        while status == "IN_PROGRESS":
            time.sleep(5)
            response = self.extractor.get_document_analysis(JobId=job_id)
            status = response["JobStatus"]
            yield status, response.get("StatusMessage", None)

    def get_job_results(self, job_id: str):
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

    def _extract(self, town_pdf_path: str):
        job_id = self.start_job(town_pdf_path)
        for s in self.get_job_status(job_id):
            status, status_message = s
            if status == "FAILED":
                print(
                    f"Job {job_id} on file {town_pdf_path} FAILED. Reason: {status_message}"
                )
            elif status == "SUCCEEDED":
                result = list(self.get_job_results(job_id))
                target_path = os.path.join(
                    self.config.data_output_dir,
                    self.config.target_state,
                    "extract_dataset",
                    os.path.basename(town_pdf_path.replace(".pdf", ".json")),
                )
                with open(target_path, "w", encoding="utf-8") as f:
                    json.dump(result, f)
                print(f"Job {job_id} on file {town_pdf_path} SUCCEEDED.")

    def extract(self, state_all_towns_names: list[str]):
        state_all_towns_zoning_files = [
            f"zoning/{self.config.target_state}/zoning-{town}.pdf"
            for town in state_all_towns_names
        ]
        thread_map(self._extract, state_all_towns_zoning_files)

    def process_town(self, town: str):
        """
        Inputs:
            town (string): name of town whose text data to import
        Returns: TODO
        """

        page_output_dir = os.path.join(
            self.config.data_output_dir,
            self.config.target_state,
            "extract_page_dataset",
        )

        os.makedirs(page_output_dir, exist_ok=True)

        filename = os.path.join(
            self.config.data_output_dir,
            self.config.target_state,
            "extract_dataset",
            f"{town}-zoning-code.json",
        )

        with open(filename, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding {filename}")
                return None

        extract_blocks = [b for d in data for b in d["Blocks"]]

        entities = Entities([], set(), {})
        rows = []
        for w in tqdm.tqdm(extract_blocks):
            if w["BlockType"] in ["LINE", "WORD", "CELL", "MERGED_CELL"]:
                e = Entity(
                    w["Id"],
                    w.get("Text", ""),
                    w["BlockType"],
                    self.collect_relations(w),
                    (
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
                            "Town": f"{self.config.target_state}-{town}",
                            "Page": w["Page"] - 1,
                            "Text": str(entities),
                        }
                    )
                entities = Entities([], set(), {})
            elif w["BlockType"] == "TABLE":
                pass
            else:
                continue

        if len(entities.ents) > 0:
            rows.append(
                {
                    "Town": f"{self.config.target_state}-{town}",
                    "Page": w["Page"],
                    "Text": str(entities),
                }
            )

        page_output_path = os.path.join(
            page_output_dir,
            f"{town}-zoning-code.jsonl",
        )
        with jsonlines.open(page_output_path, "w") as f:
            f.write_all(rows)

        return page_output_path

    def collect_relations(self, w):
        rels = w["Relationships"] if "Relationships" in w else []
        ids = []
        for r in rels if rels else []:
            for id in r["Ids"]:
                ids.append(id)
        return ids

    def linearize(self, dataset: Dataset):
        entities = Entities([], set(), {})
        rows = []
        for w in tqdm(dataset):
            if w["BlockType"] in ["LINE", "WORD", "CELL", "MERGED_CELL"]:
                e = Entity(
                    w["Id"],
                    w.get("Text", ""),
                    w["BlockType"],
                    self.collect_relations(w),
                    (w["RowIndex"], w["ColumnIndex"]),
                )
                entities.add(e)
            elif w["BlockType"] == "PAGE":
                rows.append(
                    {"Town": w["Town"], "Page": w["Page"], "Text": str(entities)}
                )
                entities = Entities([], set(), {})
            elif w["BlockType"] == "TABLE":
                pass
            else:
                continue
        return Dataset.from_list(rows)

    def post_extract(self, state_all_towns_names: list[str]):
        if self.config.target_state == "all":
            raise NotImplementedError(
                "Post-extraction for all states not yet implemented."
            )

        state_page_data_files = [
            path
            for path in process_map(self.process_town, state_all_towns_names)
            if path is not None
        ]

        page_dataset = load_dataset("json", data_files=state_page_data_files)

        hf_dataset_path = os.path.join(
            self.config.data_output_dir,
            self.config.target_state,
            "hf_dataset",
        )
        page_dataset.save_to_disk(hf_dataset_path)

        if self.config.extract.hf_dataset.publish_dataset:
            page_dataset.push_to_hub(
                self.config.extract.hf_dataset.name,
                private=self.config.extract.hf_dataset.private,
            )
