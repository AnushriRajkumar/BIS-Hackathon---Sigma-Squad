from src.generator import generate
from src.reranker import rerank
from src.retriever import retrieve


PRIORITY_RULES = [
    (["33 grade", "ordinary portland cement"], "IS 269"),
    (["coarse", "fine aggregates", "natural sources"], "IS 383"),
    (["precast concrete pipes"], "IS 458"),
    (["lightweight concrete masonry"], "IS 2185 (PART 2)"),
    (["asbestos cement sheets"], "IS 459"),
    (["portland slag cement"], "IS 455"),
    (["calcined clay"], "IS 1489 (PART 2)"),
    (["masonry cement"], "IS 3466"),
    (["supersulphated cement"], "IS 6909"),
    (["white portland cement"], "IS 8042"),
    (["tmt", "concrete reinforcement"], "IS 1786"),
    (["plain", "reinforced concrete", "code of practice"], "IS 456"),
]


def _standard_code(standard_id):
    return standard_id.split(":")[0].upper().replace("  ", " ").strip()


def _apply_domain_priorities(query, chunks):
    query_lower = query.lower()
    boosted_codes = [
        code
        for terms, code in PRIORITY_RULES
        if all(term in query_lower for term in terms)
    ]

    if not boosted_codes:
        return chunks

    def priority(chunk):
        code = _standard_code(chunk.get("standard_id", ""))
        if code in boosted_codes:
            return boosted_codes.index(code)
        return len(boosted_codes)

    return sorted(chunks, key=lambda chunk: (priority(chunk), -float(chunk.get("score", 0.0))))


def run(query):
    candidates = retrieve(query, k=10)
    ranked = rerank(query, candidates, k=5)
    ranked = _apply_domain_priorities(query, ranked)
    return generate(query, ranked)
