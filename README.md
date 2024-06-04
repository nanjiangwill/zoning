# zoning


## Download PDFs and store in S3

download all zoning PDFs and store in S3 in the format
`zoning/<target_state>/zoning-<town_name>.pdf`


## Data Extraction

For each state, create a new dir in data
`mkdir -p <state_name>`

### Running OCR
WIP
```
cp .env.example .env
fill
```

After running OCR/extraction, it will generate data inside `data/<state_name>/extract_dataset` and gather information from same page to be one data sample, which is stored in `data/<state_name>/extract_page_dataset`. `data/<state_name>/hf_dataset` is same to `data/<state_name>/extract_page_dataset` and can be read by `load_datasets` directly

## Indexing

### Running an ElasticSearch Cluster locally

A Docker Compose setup for running a full ElasticSearch stack with Logstash and
Kibana is provided by the Docker Organization. This is the easiest way to run
ElasticSearch locally, but it requires having Docker available on your machine.

If you have Docker available, you can clone the repository and start the cluster
by running:

```
git clone https://github.com/maxdumas/awesome-compose
cd awesome-compose/elasticsearch-logstash-kibana
docker compose up
```
The initial startup may take some time.

### Indexing with ES

WIP

## Search

## LLM

## Error Analysis
