import json
import os
import time
from typing import Generator, Tuple

import boto3
from tqdm.contrib.concurrent import thread_map

from zoning.class_types import OCRConfig
from zoning.ocr.base_extractor import Extractor


class TextractExtractor(Extractor):
    def __init__(self, ocr_config: OCRConfig):
        super().__init__(ocr_config)
        if self.ocr_config.run_ocr:
            self.extractor = boto3.client("textract")

    def start_job(self, s3_bucket_name: str) -> str:
        """Runs Textract's StartDocumentAnalysis action and specifies an s3
        bucket to dump output."""
        response = self.extractor.start_document_analysis(
            DocumentLocation={
                "S3Object": {
                    "Bucket": self.ocr_config.input_document_s3_bucket,
                    "Name": s3_bucket_name,
                }
            },
            FeatureTypes=self.ocr_config.feature_types,
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

    def _extract(self, ocr_entity: OCREntity) -> None:
        if self.ocr_config.pdf_name_prefix_in_s3_bucket:
            s3_bucket_name = (
                self.ocr_config.pdf_name_prefix_in_s3_bucket + ocr_entity.pdf_file
            )
        else:
            s3_bucket_name = ocr_entity.pdf_file
        job_id = self.start_job(s3_bucket_name)
        for s in self.get_job_status(job_id):
            status, status_message = s
            if status == "FAILED":
                print(
                    f"Job {job_id} on file {ocr_entity.pdf_file} FAILED. Reason: {status_message}"
                )
            elif status == "SUCCEEDED":
                result = list(self.get_job_results(job_id))
            
                with open(ocr_entity.ocr_results_file, "w", encoding="utf-8") as f:
                    json.dump(result, f)
                print(
                    f"Job {job_id} on file {ocr_entity.pdf_file} SUCCEEDED. Write to {ocr_entity.ocr_results_file}"
                )

    def process_files_and_write_output(self, config) -> None:
        if self.ocr_config.run_ocr:
            thread_map(self._extract, ocr_entities.ocr_entities)
        assert len(os.listdir(ocr_entities.ocr_results_dir)) > 0, "No OCR results found"
