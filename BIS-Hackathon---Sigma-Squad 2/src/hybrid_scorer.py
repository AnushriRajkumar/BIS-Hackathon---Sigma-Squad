"""
hybrid_scorer.py (FINAL FIXED VERSION)

✔ Embedding + BM25 hybrid scoring
✔ BM25 normalization
✔ Category boosting
✔ Token overlap boost (important for BIS IDs)
✔ Safe score clipping
✔ Optimized for Hit@3 + MRR@5
"""

from __future__ import annotations

import re
import numpy as np
from typing import List, Dict


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
EMBED_WEIGHT = 0.7
BM25_WEIGHT = 0.3

CATEGORY_BOOST = 0.15
CATEGORY_PENALTY = -0.05

OVERLAP_BOOST = 0.1

MIN_SCORE = 0.0
MAX_SCORE = 1.0


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)

    if max_s - min_s == 0:
        return [0.0] * len(scores)

    return [(s - min_s) / (max_s - min_s) for s in scores]


def extract_tokens(text: str) -> set:
    text = text.lower()

    # normal words
    words = set(re.findall(r"[a-z]{2,}", text))

    # BIS standard IDs like IS 269
    ids = set(
        re.sub(r"\s+", "", m.lower())
        for m in re.findall(r"\bIS[\s:]*\d+\b", text)
    )

    return words | ids


def overlap_score(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────
def compute_scores(
    query: str,
    
    candidates: List[Dict],
    cosine_scores: List[float],
    query_category: str | None
) -> List[Dict]:

    # ───── Normalize BM25 if present ─────
    bm25_list = [c.get("bm25_score") for c in candidates]

    if any(score is not None for score in bm25_list):
        bm25_values = [
            score if score is not None else 0.0 for score in bm25_list
        ]
        bm25_norm = normalize_scores(bm25_values)
    else:
        bm25_norm = [None] * len(candidates)

    # ───── Token sets ─────
    query_tokens = extract_tokens(query)

    results = []

    for i, c in enumerate(candidates):

        emb_score = float(cosine_scores[i])

        # ───── Hybrid base score ─────
        if bm25_norm[i] is not None:
            base = EMBED_WEIGHT * emb_score + BM25_WEIGHT * bm25_norm[i]
        else:
            base = emb_score

        # ───── Category boost ─────
        category = c.get("category")
        if query_category:
            if category == query_category:
                base += CATEGORY_BOOST
            elif category:
                base += CATEGORY_PENALTY

        # ───── Token overlap boost ─────
        text = c.get("text", "")
        cand_tokens = extract_tokens(text)

        overlap = overlap_score(query_tokens, cand_tokens)
        base += overlap * OVERLAP_BOOST

        # ───── Clip score ─────
        final_score = float(np.clip(base, MIN_SCORE, MAX_SCORE))

        results.append({
            **c,
            "score": final_score
        })

    return results