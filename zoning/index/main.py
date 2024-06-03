import os
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_from_disk, DatasetDict
from indexer import *
from typing import cast
from elasticsearch import Elasticsearch


@hydra.main(version_base=None, config_path="../../config/index", config_name="base")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    match config.index.method:
        case "keyword":
            indexer = KeywordIndexer(config)
        case "embdeding":
            indexer = EmbeddingIndexer(config)
        case _:
            raise ValueError(f"Extractor {config.extract.name} not implemented")

    # TODO, merge output_dir and target_state to global variable
    dataset_path = os.path.join(
        config.index.output_dir, config.index.target_state, "hf_dataset"
    )
    dataset = load_from_disk(dataset_path)
    dataset = cast(DatasetDict, dataset)

    es = Elasticsearch(config.index.es_endpoint)

    # TODO, currently we do not split the dataset, we index the whole dataset, but load_dataset need to specify train/test, so we store everything in train
    indexer.index(es, dataset=dataset["train"])
    # indexer.index(dataset=dataset['train'])


if __name__ == "__main__":
    main()


# from typing import cast

# from datasets import load_from_disk, DatasetDict, Dataset
# from elasticsearch import Elasticsearch
# import tiktoken
# from tqdm.contrib.concurrent import thread_map

# from ..utils import get_project_root

# DATA_ROOT = get_project_root() / "data"

# input_hf_dataset_path = DATA_ROOT / "hf_text_dataset"

# enc = tiktoken.encoding_for_model("text-davinci-003")

# # Load data set
# es = Elasticsearch("http://localhost:9200")  # default client


# def get_town_data(data: Dataset, town: str):
#     "Return a dataset for a town zoning code with embedding lookups"
#     d = data.filter(lambda x: x["Town"] == town)
#     return d


# def index_dataset(d, index_name):
#     es.indices.delete(index=index_name, ignore=[400, 404])

#     for page in d.index:
#         text = ""
#         # Include 10 pages of forward context in the index
#         for j in range(10):
#             if page + j not in d.index:
#                 continue
#             text += f"\nNEW PAGE {page + j - 1}\n" + d.loc[page + j]["Text"]

#         # Truncate to 2000 tokens
#         text = enc.decode(enc.encode(text)[:2500])
#         es.index(index=index_name, id=page, document={"Page": page, "Text": text}, request_timeout=30)


# def main(st=None):
#     ds = cast(DatasetDict, load_from_disk(input_hf_dataset_path))

#     for split in ds.keys():
#         print(f"Processing {split} split...")
#         df = ds[split].to_pandas().set_index(["Town", "Page"])
#         if st:
#             st.write(f"Processing {split} split...")
#             st.write(df)
#         towns = set(df.index.get_level_values(0))
#         thread_map(lambda town: index_dataset(df.loc[town], town), towns)


# # Make index for every town.
# if __name__ == "__main__":
#     main()
