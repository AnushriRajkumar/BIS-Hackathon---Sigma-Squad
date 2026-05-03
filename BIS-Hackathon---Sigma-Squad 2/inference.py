import argparse
import json
import time
from pathlib import Path

from src.pipeline import run


def extract_ids(result):
    ids = []
    seen = set()

    for item in result:
        standard_id = item.get("standard_id", "").strip()
        if standard_id and standard_id not in seen:
            ids.append(standard_id)
            seen.add(standard_id)

    return ids[:5]


def main(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    outputs = []

    # Warm up the embedding model and FAISS index before per-query timing.
    run("warmup query for cement steel concrete aggregates")

    for index, item in enumerate(data):
        query = item.get("query", "")
        query_id = item.get("id", str(index))

        start_time = time.perf_counter()
        result = run(query)
        latency = time.perf_counter() - start_time

        output_item = {
            "id": query_id,
            "retrieved_standards": extract_ids(result),
            "latency_seconds": round(latency, 4),
        }
        if "expected_standards" in item:
            output_item["expected_standards"] = item["expected_standards"]

        outputs.append(output_item)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args.input, args.output)
