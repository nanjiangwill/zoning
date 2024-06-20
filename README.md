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
Fill `.env` with corresponding API keys.
dotenv will not override the env variable if it's already set.

## Stage 1 - OCR

### Input Data Location

Default: `data/<state_name>/pdfs` 

### Command

Run `python -m zoning.ocr --config-name <state_name>`
 
### Output Data Location 

Default: `data/<state_name>/ocr`

## Stage 2 - Indexing

### Input Data Location

Default: `data/<state_name>/ocr` 

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

## Search and LLM Inference

`python zoning/llm_inference.py <state_name>`
Warning: this is different to `--config-name <state_name>` in previous code
 because `Typer` does not work well with `omegaconf`

You will get search results and LLM inference results in `result_output_dir/<state_name>/<experiment_name>`.

## Scoring

`python zoning/score.py --config-name <state_name>`

This code will return metrics in `result_output_dir/<state_name>/<experiment_name>`

## Visulization / Error Analysis

run `streamlit run viz/viz.py` and use larger resolusion.

Select files in `result_output_dir` and end with `_with_ground_truth.json`

## Notes for developers

run the following code for type check/format check/etc before commit

```bash
pre-commit autoupdate
pre-commit run --all-files
```

## Supported Config

WIP
