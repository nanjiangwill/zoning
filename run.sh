#!/bin/bash

CONFIG_NAME=$1

echo "Running config: $CONFIG_NAME"

# Stage 1: OCR
# python -m zoning.ocr --config-name $CONFIG_NAME

# Stage 2: Format OCR
python -m zoning.format_ocr --config-name $CONFIG_NAME

# Stage 3: Index
python -m zoning.index --config-name $CONFIG_NAME

# Stage 4: Search
python -m zoning.search --config-name $CONFIG_NAME

# Stage 5: Prompt
python -m zoning.prompt --config-name $CONFIG_NAME

# Stage 6: LLM
python -m zoning.llm $CONFIG_NAME

# Stage 7: Normalization
python -m zoning.normalization --config-name $CONFIG_NAME

# Stage 8: Evaluation
python -m zoning.eval --config-name $CONFIG_NAME
