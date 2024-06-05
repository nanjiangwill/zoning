# zoning

Introduction WIP

ONLY WORK FOR ONE STATE NOW

## Download PDFs and store in S3 (Optional)

Download all zoning PDFs and store in S3 in the format
`zoning/<target_state>/zoning-<town_name>.pdf`

## Config

check [Supported Config below](#supported-config) for available configs

```bash
cp config/base.yaml config/<state_name>.yaml
cp .env.example .env
```

Change the config. You can refer to `config/connecticut.yaml` and
fill `.env` with corresponding API keys.

## Data Extraction

### Step 1

Create a new dir in data `mkdir -p data/<state_name>/pdfs` and move all related
 pdfs to this folder.

### Step 2

Option 1: Not redo textract OCR

- Ask for extracted json files and store them in `data/<state_name>/extract_dataset`
- Run `python zoning/extract.py --config-name <state_name>`
 with `extraction.run_ocr: false`

Option 2: Redo textract OCR

- Run `python zoning/extract.py --config-name <state_name>`
 with `extraction.run_ocr: true`
- It will generate data inside `data/<state_name>/extract_dataset`

After running OCR/extraction, it will gather information from same page and make
 it to one data sample, which is stored in `data/<state_name>/extract_page_dataset`.
  `data/<state_name>/hf_dataset` is same to `data/<state_name>/extract_page_dataset`
   and can be read by `load_datasets` directly.

## Indexing

### Running an ElasticSearch Cluster locally

A Docker Compose setup for running a full ElasticSearch stack with Logstash and
Kibana is provided by the Docker Organization. This is the easiest way to run
ElasticSearch locally, but it requires having Docker available on your machine.

If you have Docker available, you can clone the repository and start the cluster
by running:

```bash
git clone https://github.com/maxdumas/awesome-compose
cd awesome-compose/elasticsearch-logstash-kibana
docker compose up
```

The initial startup may take some time.

### Run indexing code

`python zoning/index.py --config-name <state_name>`

## Evaluation

`python zoning/eval.py <state_name>`
Warning: this is different to `--config-name <state_name>` in previous code
 because `Type`r does not work well with `omegaconf`

You will get result in `result_output_dir/<state_name>`

- `result_output_dir/<state_name>/metrics.json` for accuracy/F1/etc score
- `result_output_dir/<state_name>/pr_answers.csv` for samples that got right
 with larger search scope
- `result_output_dir/<state_name>/wrong_answers.csv` for error analysis

## Visulization / Error Analysis

with `result_output_dir/<state_name>/wrong_answers.csv`, there are some
 exisiting code in `viz/viz.ipynb` for error analysis

## Notes for developers

run the following code for type check/format check/etc before commit

```bash
pre-commit autoupdate
pre-commit run --all-files
```

## Supported Config

WIP
