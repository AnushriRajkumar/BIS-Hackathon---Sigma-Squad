# BIS Standards Recommendation Engine

Submission for the Bureau of Indian Standards x Sigma Squad AI Hackathon.

Theme: Accelerating MSE Compliance - Automating BIS Standard Discovery

## Overview

This project recommends relevant BIS standards from a product description using a retrieval-augmented pipeline focused on building materials such as cement, steel, concrete, and aggregates.

Pipeline:

```text
Product query -> Retriever -> Reranker -> Generator -> Top BIS standards
```

## Project Structure

```text
project/
├── src/
│   ├── retriever.py
│   ├── reranker.py
│   ├── generator.py
│   ├── pipeline.py
│   ├── ingest.py
│   ├── index.py
│   ├── hybrid_scorer.py
│   └── query_normalizer.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── indexes/
├── inference.py
├── eval_script.py
├── public_test_set.json
└── requirements.txt
```

## How It Works

- `src/ingest.py`: extracts BIS PDF text and creates structured standard records.
- `src/index.py`: builds the FAISS vector index.
- `src/retriever.py`: retrieves the top candidate BIS chunks.
- `src/reranker.py`: reranks candidates for better top-3 precision.
- `src/generator.py`: produces grounded recommendation objects.
- `src/pipeline.py`: integrates retriever, reranker, and generator.
- `inference.py`: mandatory judge entrypoint.

## Setup

```bash
pip install -r requirements.txt
```

The repository already includes the processed dataset and FAISS index. To rebuild them:

```bash
python src/ingest.py
python src/index.py
```

## Run Inference

```bash
python inference.py --input public_test_set.json --output team_results.json
```

## Evaluate

```bash
python eval_script.py --results team_results.json
```

Public test results:

```text
Hit Rate @3 : 100.00%
MRR @5      : 1.0000
Avg Latency : 0.03 sec
```

## Team

- Sara Y
- Anushri Rajkumar
- Shree Rathina Kumar
