"""
reranker.py (NO TORCH VERSION)

✔ Works on Python 3.14
✔ No sentence-transformers
✔ Uses TF-IDF + cosine similarity
✔ Still supports hybrid scoring + boosts
"""

from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .hybrid_scorer import compute_scores


def normalize_query(query: str):
    query = query.lower()

    if any(x in query for x in ["cement", "opc", "ppc"]):
        category = "cement"
    elif any(x in query for x in ["steel", "rod", "bar"]):
        category = "steel"
    elif any(x in query for x in ["concrete", "rcc"]):
        category = "concrete"
    elif any(x in query for x in ["aggregate", "sand"]):
        category = "aggregate"
    else:
        category = None

    return {
        "normalized_query": query,
        "category": category
    }


def rerank(query: str, candidates: List[Dict], k: int = 5):

    if not candidates:
        return []

    # ───── Normalize query ─────
    q = normalize_query(query)
    query_text = q["normalized_query"]
    query_category = q["category"]

    # ───── Prepare texts ─────
    texts = [query_text] + [c.get("text", "") for c in candidates]

    # ───── TF-IDF ─────
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    query_vec = tfidf[0]
    doc_vecs = tfidf[1:]

    # ───── Cosine similarity ─────
    cosine_scores = cosine_similarity(query_vec, doc_vecs)[0]

    # ───── Hybrid scoring ─────
    scored = compute_scores(
        query=query_text,
        
        candidates=candidates,
        cosine_scores=cosine_scores,
        query_category=query_category
    )

    # ───── Sort ─────
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)

    # ───── Deduplicate ─────
    seen = set()
    unique = []
    for c in scored:
        sid = c.get("standard_id")
        if sid and sid not in seen:
            unique.append(c)
            seen.add(sid)

    return unique[:k]