import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("data/indexes")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_index = None
_records = None


DOMAIN_EXPANSIONS = {
    "tmt": ["thermo mechanically treated", "high strength deformed", "reinforcing bars", "reinforcement"],
    "rebar": ["reinforcing bars", "reinforcement", "deformed steel bars"],
    "rebars": ["reinforcing bars", "reinforcement", "deformed steel bars"],
    "opc": ["ordinary portland cement"],
    "ppc": ["portland pozzolana cement"],
    "rcc": ["reinforced concrete", "plain and reinforced concrete"],
}


def load_index():
    global _model, _index, _records

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    if _index is None:
        index_path = INDEX_DIR / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError("Missing FAISS index. Run: python3 src/index.py")
        _index = faiss.read_index(str(index_path))

    if _records is None:
        records_path = INDEX_DIR / "records.pkl"
        if not records_path.exists():
            raise FileNotFoundError("Missing records file. Run: python3 src/index.py")
        with records_path.open("rb") as f:
            _records = pickle.load(f)


def _normalize_query(query: str) -> str:
    expanded = query.lower()
    for term, additions in DOMAIN_EXPANSIONS.items():
        if term in expanded:
            expanded = expanded + " " + " ".join(additions)
    return expanded


def _tokens(text: str):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_score(query: str, record):
    query_tokens = _tokens(_normalize_query(query))
    record_text = " ".join(
        [
            record.get("standard_id", ""),
            record.get("title", ""),
            record.get("category", ""),
            record.get("text", "")[:2500],
        ]
    )
    record_tokens = _tokens(record_text)

    if not query_tokens:
        return 0.0

    overlap = len(query_tokens & record_tokens) / len(query_tokens)
    phrase_bonus = 0.0
    normalized_record = record_text.lower()

    for phrase in [
        "plain and reinforced concrete",
        "ordinary portland cement",
        "coarse and fine aggregates",
        "high strength deformed steel bars",
        "concrete reinforcement",
    ]:
        if phrase in _normalize_query(query) and phrase in normalized_record:
            phrase_bonus += 0.15

    return min(1.0, overlap + phrase_bonus)


def retrieve(query: str, k: int = 10):
    load_index()

    query_embedding = _model.encode(
        [query],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(query_embedding, dtype="float32")
    candidate_count = min(max(k * 5, 30), len(_records))
    scores, indices = _index.search(query_embedding, candidate_count)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        record = _records[idx]
        keyword_score = _keyword_score(query, record)
        final_score = (0.75 * float(score)) + (0.25 * keyword_score)
        results.append(
            {
                "text": record.get("text", ""),
                "standard_id": record.get("standard_id", ""),
                "title": record.get("title", ""),
                "category": record.get("category", ""),
                "score": final_score,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:k]
