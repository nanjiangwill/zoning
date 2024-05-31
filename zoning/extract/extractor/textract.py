from base import *
from omegaconf import DictConfig
import boto3
import time
import json
import os
import pandas as pd
import pyarrow as pa
from tqdm.contrib.concurrent import process_map
from datasets import load_dataset, Dataset, DatasetDict
import tqdm
import numpy as np
import openai

SCHEMA = pa.schema(
    [
        pa.field("Town", pa.string()),
        pa.field("BlockType", pa.string()),
        pa.field("ColumnIndex", pa.int32(), nullable=True),
        pa.field("ColumnSpan", pa.int32(), nullable=True),
        pa.field("Confidence", pa.float64(), nullable=True),
        # pa.field("EntityTypes", pa.list_(pa.string()), nullable=True),
        pa.field("Id", pa.string()),
        pa.field("Page", pa.int32()),
        pa.field(
            "Relationships",
            pa.list_(
                pa.struct(
                    [
                        pa.field("Ids", pa.list_(pa.string())),
                        pa.field("Type", pa.string()),
                    ]
                )
            ),
            nullable=True,
        ),
        pa.field("RowIndex", pa.int32(), nullable=True),
        pa.field("RowSpan", pa.int32(), nullable=True),
        pa.field("Text", pa.string(), nullable=True),
        pa.field("TextType", pa.string(), nullable=True),
        pa.field(
            "Geometry",
            pa.struct(
                [
                    pa.field(
                        "BoundingBox",
                        pa.struct(
                            [
                                pa.field("Width", pa.float64()),
                                pa.field("Height", pa.float64()),
                                pa.field("Left", pa.float64()),
                                pa.field("Top", pa.float64()),
                            ]
                        ),
                    ),
                    pa.field(
                        "Polygon",
                        pa.list_(
                            pa.struct(
                                [
                                    pa.field("X", pa.float64()),
                                    pa.field("Y", pa.float64()),
                                ]
                            )
                        ),
                    ),
                ]
            ),
        ),
    ]
)


class TextractExtractor(Extractor):
    def __init__(self, extractor_config: DictConfig):
        super().__init__(extractor_config)
        self.extractor = boto3.client("textract")
        self.schema = SCHEMA

    def start_job(self, town_pdf_path: str) -> str:
        """
        Runs Textract's StartDocumentAnalysis action and
        specifies an s3 bucket to dump output
        """
        response = self.extractor.start_document_analysis(
            DocumentLocation={
                "S3Object": {
                    "Bucket": self.config.input_document_s3_bucket,
                    "Name": town_pdf_path,
                }
            },
            FeatureTypes=self.config.feature_types,
        )

        return response["JobId"]

    def get_job_status(self, job_id: str):
        """'
        Checks whether document analysis still in progress
        """
        status: str = "IN_PROGRESS"
        while status == "IN_PROGRESS":
            time.sleep(5)
            response = self.extractor.get_document_analysis(JobId=job_id)
            status = response["JobStatus"]
            yield status, response.get("StatusMessage", None)

    def get_job_results(self, job_id: str):
        """
        If document analysis complete, runs Textract's GetDocumentAnalysis action
        and pulls JSON results to be stored in s3 bucket designated above
        """
        response = self.extractor.get_document_analysis(JobId=job_id)
        nextToken = response.get("NextToken", None)
        yield response

        while nextToken is not None:
            response = self.extractor.get_document_analysis(
                JobId=job_id, NextToken=nextToken
            )
            nextToken = response.get("NextToken", None)
            yield response

    def extract(self, town_pdf_path: str):
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
                    self.config.output_path,
                    self.config.target_state,
                    "extract_dataset",
                    os.path.basename(town_pdf_path.replace(".pdf", ".json")),
                )
                with target_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f)
                print(f"Job {job_id} on file {town_pdf_path} SUCCEEDED.")

    def import_town(self, town: str):
        """
        Inputs:
            town (string): name of town whose text data to import
        Returns: pandas dataframe of cleaned/combined JSONs for a given document with all Textract information
        """

        filename = os.path.join(
            self.config.output_path,
            self.config.target_state,
            "extract_dataset",
            f"{town}-zoning-code.json",
        )

        with filename.open() as f:
            data = json.load(f)

        df = pd.DataFrame(
            [b for d in data for b in d["Blocks"]], columns=SCHEMA.names
        ).drop_duplicates(subset="Id")
        df["Town"] = town

        parquet_output_path = os.path.join(
            self.config.output_path,
            self.config.target_state,
            "parquet_dataset",
            f"{town}.parquet",
        )
        df.to_parquet(parquet_output_path, schema=SCHEMA)

        return parquet_output_path

    def collect_relations(self, w):
        rels = w["Relationships"]
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

    def embed(self, x):
        y = []
        for p in x["Text"]:
            if not p:
                p = " "
            if len(p.split()) > 3000:
                p = " ".join(p[:3000])
            y.append(p)
        emb = openai.Embedding.create(input=y, engine=self.config.embedding_model)
        return {
            "embeddings": [
                np.array(emb["data"][i]["embedding"]) for i in range(len(emb["data"]))
            ]
        }

    def post_extract(self):
        if self.config.target_state == "all":
            raise NotImplementedError(
                "Post-extraction for all states not yet implemented."
            )

        state_data_path = os.join(self.config.output_path, self.config.target_state)
        state_all_towns_names_path = os.join(state_data_path, "all_towns_names.json")
        with state_all_towns_names_path.open() as f:
            state_all_towns_names = json.load(f)

        state_parquet_data_files = [
            path
            for path in process_map(self.import_town, state_all_towns_names)
            if path is not None
        ]

        # TODO, this is a hack, need to fix
        dataset = load_dataset(
            "parquet", data_files={"train": state_parquet_data_files}
        )
        dataset["train"] = self.linearize(dataset["train"]).map(
            self.embed, batch_size=100, batched=True
        )
        # Hack ends here
        hf_dataset_path = os.path.join(
            self.config.output_path, self.config.target_state, "hf_dataset"
        )
        dataset.save_to_disk(hf_dataset_path)

        if self.config.hf_dataset.publish:
            dataset.push_to_hub(
                self.config.hf_dataset.name, private=self.config.hf_dataset.private
            )
