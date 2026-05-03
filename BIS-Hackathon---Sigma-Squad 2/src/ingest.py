import json
import re
from pathlib import Path

import fitz

RAW_PDF = Path("data/raw/dataset.pdf")
OUT_PATH = Path("data/processed/standards.json")


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        pages.append(f"\n\n--- PAGE {page_num} ---\n{text}")

    return "\n".join(pages)


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guess_category(text):
    lower = text.lower()

    category_terms = {
        "cement": ["cement", "opc", "ppc", "portland", "pozzolana", "clinker"],
        "steel": ["steel", "tmt", "bar", "reinforcement", "wire", "structural"],
        "concrete": ["concrete", "mix", "admixture", "mortar"],
        "aggregates": ["aggregate", "sand", "gravel", "crushed stone"],
    }

    scores = {}
    for category, terms in category_terms.items():
        scores[category] = sum(lower.count(term) for term in terms)

    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return "building_materials"

    return best_category


def normalize_standard_id(raw_id):
    raw_id = re.sub(r"\s+", " ", raw_id.upper()).strip()
    raw_id = raw_id.replace("I S", "IS")
    raw_id = re.sub(r"\(\s*PART\s*(\d+)\s*\)", r"(PART \1)", raw_id)
    raw_id = re.sub(r"\s*:\s*", ": ", raw_id)
    raw_id = re.sub(r"\s+\(", " (", raw_id)
    return raw_id.strip()


def parse_standards(full_text):
    # Matches examples like:
    # IS 456
    # IS 1786
    # IS 383 (Part 1)
    # IS 4031 (Part 5)
    pattern = re.compile(
        r"(?m)^IS\s*\d{2,5}(?:\s*\(PART\s*\d+\))?\s*:\s*\d{4}",
        flags=re.IGNORECASE,
    )

    matches = list(pattern.finditer(full_text))
    records = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

        block = clean_text(full_text[start:end])
        if len(block) < 80:
            continue

        standard_id = normalize_standard_id(match.group(0))
        lines = [line.strip() for line in block.splitlines() if line.strip()]

        title = ""
        for line in lines[1:8]:
            if not line.upper().startswith("IS ") and len(line) > 8:
                title = line
                break

        records.append(
            {
                "standard_id": standard_id,
                "title": title,
                "category": guess_category(block),
                "text": block,
            }
        )

    # De-duplicate by standard_id, keeping the longest text block.
    best_by_id = {}
    for record in records:
        existing = best_by_id.get(record["standard_id"])
        if existing is None or len(record["text"]) > len(existing["text"]):
            best_by_id[record["standard_id"]] = record

    return list(best_by_id.values())


def main():
    if not RAW_PDF.exists():
        raise FileNotFoundError(f"Could not find {RAW_PDF}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {RAW_PDF}")
    full_text = extract_text(RAW_PDF)
    full_text = clean_text(full_text)

    records = parse_standards(full_text)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} standards to {OUT_PATH}")


if __name__ == "__main__":
    main()
