import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = Path("data/processed/standards.json")
INDEX_DIR = Path("data/indexes")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Run src/ingest.py first.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with DATA_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    texts = []
    for record in records:
        text = " ".join(
            [
                record.get("standard_id", ""),
                record.get("title", ""),
                record.get("category", ""),
                record.get("text", ""),
            ]
        )
        texts.append(text)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts)} records")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with (INDEX_DIR / "records.pkl").open("wb") as f:
        pickle.dump(records, f)

    print(f"Wrote FAISS index to {INDEX_DIR / 'faiss.index'}")
    print(f"Wrote records to {INDEX_DIR / 'records.pkl'}")


if __name__ == "__main__":
    main()
