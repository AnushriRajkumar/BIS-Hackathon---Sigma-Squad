def generate(query, chunks):
    results = []

    for chunk in chunks[:5]:
        title = chunk.get("title", "").strip()
        standard_id = chunk.get("standard_id", "").strip()
        text = chunk.get("text", "").strip()

        rationale_source = title or text[:160]
        reason = (
            f"Relevant because the retrieved BIS context discusses {rationale_source} "
            f"in relation to the product query."
        )

        results.append(
            {
                "standard_id": standard_id,
                "title": title,
                "reason": reason,
                "score": float(chunk.get("score", 0.0)),
            }
        )

    return results
