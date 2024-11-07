import json
import os
import time
from functools import partial
from typing import Generator, Tuple

import boto3
from tqdm.contrib.concurrent import thread_map

from zoning.class_types import OCRConfig
from zoning.ocr.base_extractor import Extractor
from zoning.utils import target_name, target_pdf


class TextractExtractor(Extractor):
    def __init__(self, ocr_config: OCRConfig):
        super().__init__(ocr_config)
        if self.ocr_config.run_ocr:
            self.extractor = boto3.client(
                "textract", region_name=self.ocr_config.textract_region_name
            )

    def start_job(self, pdf_file: str) -> str:
        """Runs Textract's StartDocumentAnalysis action and specifies an s3
        bucket to dump output."""
        response = self.extractor.start_document_analysis(
            DocumentLocation={
                "S3Object": {
                    "Bucket": self.ocr_config.input_document_s3_bucket,
                    "Name": pdf_file,
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

    def extract(self, ocr_dir: str, target: str) -> None:
        pdf_file = target_pdf(target, self.ocr_config.pdf_name_prefix_in_s3_bucket)
        ocr_file = target_name(target, ocr_dir)

        job_id = self.start_job(pdf_file)

        print(f"Job {job_id} on town {target}")
        for s in self.get_job_status(job_id):
            status, status_message = s
            if status == "FAILED":
                print(f"Job {job_id} on town {target} FAILED. Reason: {status_message}")
            elif status == "SUCCEEDED":
                result = list(self.get_job_results(job_id))

                with open(ocr_file, "w") as f:
                    json.dump(result, f)
                print(f"Job {job_id} on town {target} SUCCEEDED. Write to {ocr_file}")

    def process_files_and_write_output(self, target_towns: str, ocr_dir: str) -> None:
        if self.ocr_config.run_ocr:
            target_towns = json.load(open(target_towns))

            # Textract only allows 10 concurrent jobs
            thread_map(partial(self.extract, ocr_dir), target_towns, max_workers=10)

            # for target in tqdm.tqdm(target_towns):
            #     print("Running Textract on town: ", target)
            #     self.extract(ocr_dir, target)

        assert len(os.listdir(ocr_dir)) > 0, "No OCR results found"
