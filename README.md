
# zoning

Introduction WIP

ONLY WORK FOR ONE STATE NOW

## Download PDFs and store in S3 (Optional)

Download all zoning PDFs and store in S3 in the format

`zoning/<target_state>/zoning-<town_name>.pdf`

## Config

check [Supported Config below](#supported-config) for available configs

```bash

cp  config/base.yaml  config/<experiment_name>.yaml

cp  .env.example  .env

```

Change the config. You can refer to `config/connecticut.yaml` and

Fill `.env` with corresponding API keys.

dotenv will not override the env variable if it's already set.

## Run all stages

### Command

`python run.py <experiment_name>`

## Stage 1 - OCR

### Input Data Location

Default: `data/<state_name>/pdf`

### Command

Run `python -m zoning.ocr --config-name <state_name>`

### Output Data Location

Default: `data/<state_name>/ocr`

### Notes

To skip this stage, ask for ocr results and put them in `data/<state_name>/ocr`

## Stage 2 - Format OCR

### Input Data Location

Default: `data/<state_name>/ocr`

### Command

Run `python -m zoning.format_ocr --config-name <state_name>`

### Output Data Location

Default: `results/<experiment_name>/format_ocr`

## Stage 3: Indexing

### Running an ElasticSearch Cluster locally

A Docker Compose setup for running a full ElasticSearch stack with Logstash and

Kibana is provided by the Docker Organization. This is the easiest way to run

ElasticSearch locally, but it requires having Docker available on your machine.

If you have Docker available, you can clone the repository and start the cluster

by running:

```bash

git  clone  https://github.com/maxdumas/awesome-compose

cd  awesome-compose/elasticsearch-logstash-kibana

docker  compose  up

```

The initial startup may take some time.

To all clear existing entries, use

`curl -X DELETE 'http://localhost:9200/_all'`

### Input Data Location

Default: `data/<state_name>/format_ocr`

### Command

Run `python -m zoning.index --config-name <state_name>`

### Output Data Location

Default: `None`

## Stage 4: Search

### Input Data Location

Default: `None`

### Command

Run `python -m zoning.search --config-name <state_name>`

### Output Data Location

Default: `data/<state_name>/search`

## Stage 5: Prompt

### Input Data Location

Default: `data/<state_name>/search`

### Command

Run `python -m zoning.prompt --config-name <state_name>`

### Output Data Location

Default: `data/<state_name>/prompt`

## Stage 6: LLM Infernence

### Input Data Location

Default: `data/<state_name>/prompt`

### Command

Run `python -m zoning.llm  <state_name>`

Warning: this is different to `--config-name <state_name>` in previous code

because `Typer` does not work well with `omegaconf`

### Output Data Location

Default: `data/<state_name>/llm`

## Stage 7: Normalization

### Input Data Location

Default: `data/<state_name>/search`

### Command

Run `python -m zoning.normalization --config-name <state_name>`

### Output Data Location

Default: `data/<state_name>/normalization`

## Stage 8: Eval

### Input Data Location

Default: `data/<state_name>/normalization`

### Command

`python -m zoning.eval --config-name <state_name>`

### Output Data Location

Default: `data/<state_name>/eval`

## Stage 9 Visulization / Error Analysis

### Input Data Location

Default: `data/<state_name>/eval`

### Command

- run `streamlit run viz/viz.py` and use larger resolusion.
- Select files in input data location

## Notes for developers

run the following code for type check/format check/etc before commit

```bash

pre-commit  autoupdate

pre-commit  run  --all-files

```

## Supported Config

WIP
