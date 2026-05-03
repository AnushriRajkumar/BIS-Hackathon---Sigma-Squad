"""
query_normalizer.py
════════════════════════════════════════════════════════════════════════════════
Query Normalization Module for BIS Standards RAG System (Person B).

Responsibilities
────────────────
  • Clean and normalize raw user queries into precise technical terms
  • Detect the BIS building-material category (cement / steel / concrete /
    aggregates / general)
  • Expand synonyms so downstream embedding and BM25 coverage improves
  • Return a structured dict consumed by the reranker

Design notes
────────────
  • Pure-Python, zero external model calls → sub-millisecond runtime
  • All dictionaries are domain-specific to BIS SP 21 Building Materials
  • Category detection uses priority-ordered keyword matching so
    ambiguous queries ("cement concrete mix") resolve to the most
    specific category first

Author : Person B – Reranking & Query Intelligence
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNONYM / ALIAS DICTIONARY  (BIS SP-21 Building Materials domain)
# ─────────────────────────────────────────────────────────────────────────────

SYNONYM_MAP: dict[str, list[str]] = {
    # ── Cement & Binders ──────────────────────────────────────────────────────
    "ordinary portland cement": ["opc", "portland cement", "general purpose cement"],
    "portland pozzolana cement": ["ppc", "fly ash cement", "pozzolanic cement"],
    "portland slag cement": ["psc", "slag cement", "blast furnace slag cement"],
    "rapid hardening cement": ["rhc", "quick setting cement", "fast cement"],
    "sulphate resisting cement": ["src", "sulphate resistant cement", "low c3a cement"],
    "low heat cement": ["lhc", "low heat of hydration cement"],
    "white cement": ["white portland cement", "decorative cement"],
    "masonry cement": ["mortar cement", "brickwork cement"],
    "high alumina cement": ["hac", "aluminous cement", "calcium aluminate cement"],
    "super sulphated cement": ["ssc"],
    "hydrophobic cement": ["water repellent cement"],

    # ── Steel & Metals ────────────────────────────────────────────────────────
    "tmt bars": ["thermo mechanically treated bars", "tmt rebars", "tmt rebar",
                 "reinforcement bars", "rebar", "deformed bars", "hsd bars",
                 "high strength deformed bars", "fe 415", "fe 500", "fe 550",
                 "structural steel bars"],
    "mild steel": ["ms bars", "plain carbon steel", "low carbon steel"],
    "structural steel": ["rolled steel", "steel sections", "i-beam", "h-beam",
                         "channel steel", "angle steel", "ismc", "ismb"],
    "wire rod": ["steel wire", "binding wire", "hard drawn wire"],
    "galvanised steel": ["gi sheet", "galvanised iron", "zinc coated steel",
                         "hot dip galvanised"],
    "stainless steel": ["ss 304", "ss 316", "corrosion resistant steel"],
    "cold rolled steel": ["cr coil", "cold formed steel"],
    "prestressing wire": ["pc wire", "prestressed wire", "high tensile wire"],
    "pre-tensioned concrete": ["pretensioned", "pre stressed", "prestressed concrete"],

    # ── Concrete & Mixes ─────────────────────────────────────────────────────
    "ready mix concrete": ["rmc", "ready mixed concrete", "transit mix concrete",
                           "batched concrete", "ready-mix"],
    "plain cement concrete": ["pcc", "plain concrete", "unreinforced concrete"],
    "reinforced cement concrete": ["rcc", "reinforced concrete", "rc concrete"],
    "precast concrete": ["precast elements", "precast slabs", "precast panels",
                         "factory made concrete"],
    "prestressed concrete": ["psc elements", "post tensioned", "pre-stressed"],
    "high performance concrete": ["hpc", "high strength concrete", "m60 concrete",
                                  "ultra high strength concrete"],
    "lightweight concrete": ["aac blocks", "autoclaved aerated concrete",
                             "cellular concrete", "foamed concrete"],
    "self compacting concrete": ["scc", "self consolidating concrete",
                                  "flowable concrete"],
    "shotcrete": ["sprayed concrete", "gunite"],
    "concrete mix design": ["mix design", "concrete grade", "m20 m25 m30",
                            "concrete proportioning"],

    # ── Aggregates ────────────────────────────────────────────────────────────
    "coarse aggregate": ["crushed stone", "gravel", "stone chips", "aggregate",
                         "jelly", "crushed aggregate", "20mm aggregate", "10mm aggregate"],
    "fine aggregate": ["sand", "river sand", "manufactured sand", "m-sand",
                       "fine sand", "quarry dust"],
    "all-in aggregate": ["combined aggregate", "mixed grading aggregate"],
    "lightweight aggregate": ["expanded clay", "leca", "pumice",
                              "sintered fly ash aggregate"],
    "recycled aggregate": ["demolition waste aggregate", "recycled concrete aggregate",
                           "rca", "secondary aggregate"],

    # ── Bricks & Masonry ─────────────────────────────────────────────────────
    "burnt clay bricks": ["red brick", "clay brick", "common brick",
                          "first class brick", "engineering brick", "facing brick"],
    "fly ash bricks": ["fal-g brick", "pozzolanic brick", "ash brick"],
    "sand lime bricks": ["calcium silicate brick", "autoclaved sand lime brick"],
    "concrete blocks": ["hollow concrete block", "solid concrete block", "cmu",
                        "cement block", "paver block", "interlocking block"],

    # ── Tiles & Flooring ─────────────────────────────────────────────────────
    "ceramic tiles": ["vitrified tile", "floor tile", "wall tile", "porcelain tile",
                      "glazed tile", "unglazed tile"],
    "mosaic tiles": ["terrazzo", "mosaic flooring"],

    # ── Paints & Coatings ────────────────────────────────────────────────────
    "distemper": ["dry distemper", "oil bound distemper", "wall paint"],
    "enamel paint": ["oil paint", "synthetic enamel", "gloss paint"],
    "bituminous paint": ["bitumen coating", "tar paint", "anticorrosive coating"],

    # ── Water & Admixtures ───────────────────────────────────────────────────
    "water for concrete": ["mixing water", "curing water", "potable water concrete"],
    "admixture": ["superplasticizer", "plasticizer", "water reducer",
                  "accelerator", "retarder", "air entraining agent",
                  "concrete additive"],
    "fly ash": ["pulverised fuel ash", "pfa", "pond ash", "bottom ash"],
    "silica fume": ["microsilica", "condensed silica fume", "csf"],
    "ggbs": ["ground granulated blast furnace slag", "granulated slag", "ggbfs"],

    # ── Pipes & Drainage ─────────────────────────────────────────────────────
    "pvc pipes": ["polyvinyl chloride pipe", "upvc", "rigid pvc pipe"],
    "gi pipes": ["galvanised iron pipe", "ms pipe", "steel pipe"],
    "ci pipes": ["cast iron pipe", "ductile iron pipe", "di pipe"],
    "ac pipes": ["asbestos cement pipe"],
    "rcc pipes": ["concrete pipe", "hume pipe", "npc pipe"],

    # ── Timber & Wood ────────────────────────────────────────────────────────
    "plywood": ["ply board", "marine ply", "bwp plywood", "bwr plywood"],
    "particle board": ["chipboard", "wood particle board", "flakeboard"],
    "fibre cement board": ["fcb", "calcium silicate board", "fibre board"],

    # ── Glass ────────────────────────────────────────────────────────────────
    "float glass": ["sheet glass", "flat glass", "window glass"],
    "toughened glass": ["tempered glass", "safety glass"],
    "wired glass": ["fire resistant glass"],
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. CATEGORY KEYWORD DICTIONARY  (ordered by specificity)
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "cement": [
        "cement", "opc", "ppc", "psc", "portland", "clinker", "binder",
        "slag cement", "pozzolana", "masonry cement", "hydrophobic cement",
        "white cement", "rapid hardening", "sulphate resisting", "low heat",
        "high alumina", "super sulphated", "fly ash cement",
    ],
    "steel": [
        "steel", "tmt", "rebar", "rebars", "reinforcement bar", "hsd",
        "mild steel", "structural steel", "wire rod", "gi sheet", "ismc",
        "ismb", "angle iron", "channel section", "galvanised", "stainless",
        "prestressing wire", "cold rolled", "hot rolled", "ms bar",
        "deformed bar", "tor steel", "high yield",
    ],
    "concrete": [
        "concrete", "rcc", "pcc", "rmc", "ready mix", "precast", "prestressed",
        "mix design", "m20", "m25", "m30", "m40", "m50", "shotcrete",
        "lightweight concrete", "self compacting", "high performance concrete",
        "aac block", "autoclaved", "cellular concrete", "admixture",
        "superplasticizer", "fly ash concrete", "silica fume", "ggbs",
    ],
    "aggregates": [
        "aggregate", "coarse aggregate", "fine aggregate", "sand", "gravel",
        "crushed stone", "all-in aggregate", "m-sand", "manufactured sand",
        "river sand", "stone chips", "jelly", "quarry dust",
        "lightweight aggregate", "recycled aggregate", "leca", "pumice",
    ],
    "bricks": [
        "brick", "burnt clay", "fly ash brick", "sand lime brick",
        "concrete block", "hollow block", "paver", "masonry unit",
    ],
    "tiles": [
        "tile", "ceramic", "vitrified", "mosaic", "terrazzo", "flooring tile",
        "wall tile", "porcelain",
    ],
    "pipes": [
        "pipe", "pvc pipe", "gi pipe", "ci pipe", "rcc pipe", "hume pipe",
        "ductile iron", "asbestos cement pipe",
    ],
    "timber": [
        "plywood", "particle board", "timber", "fibre board", "wood",
        "chipboard", "marine ply",
    ],
    "glass": [
        "glass", "float glass", "toughened glass", "wired glass", "sheet glass",
    ],
    "paints": [
        "paint", "distemper", "enamel", "primer", "coating", "bituminous paint",
        "varnish",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. NOISE WORDS TO STRIP
# ─────────────────────────────────────────────────────────────────────────────

NOISE_WORDS: set[str] = {
    "i", "we", "our", "my", "the", "a", "an", "is", "are", "was", "were",
    "for", "of", "in", "on", "at", "to", "and", "or", "but", "with",
    "product", "material", "item", "thing", "stuff", "goods",
    "need", "want", "looking", "require", "find", "get", "use",
    "what", "which", "that", "this", "it", "be", "do", "does", "used",
    "make", "made", "about", "per", "as", "by", "from", "standard",
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. CORE NORMALIZER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def _unicode_to_ascii(text: str) -> str:
    """Strip accents and normalize unicode to ASCII-safe form."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _clean_text(text: str) -> str:
    """
    Lowercase, remove special characters, collapse whitespace.
    Keeps hyphens that join compound technical terms (e.g. 'fly-ash').
    """
    text = _unicode_to_ascii(text)
    text = text.lower()
    # Replace special chars except hyphen and alphanumeric
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _remove_noise_words(tokens: list[str]) -> list[str]:
    """Drop generic English stop/noise words from token list."""
    return [t for t in tokens if t not in NOISE_WORDS and len(t) > 1]


def _expand_synonyms(query: str) -> list[str]:
    """
    Walk the synonym map and collect all canonical terms whose
    synonyms appear in the query.  Returns a deduplicated list of
    expansion terms to append to the query.
    """
    expansions: list[str] = []
    query_lower = query.lower()

    for canonical, aliases in SYNONYM_MAP.items():
        # Check if the canonical term itself is in the query
        if canonical in query_lower:
            expansions.append(canonical)
            continue
        # Check any alias
        for alias in aliases:
            if alias in query_lower:
                expansions.append(canonical)
                break  # one match per canonical is enough

    return list(dict.fromkeys(expansions))  # preserve insertion order, deduplicate


def _detect_category(query: str) -> str:
    """
    Detect the primary BIS category from the normalised query.
    Uses priority-ordered keyword scanning; returns 'general' as fallback.
    """
    query_lower = query.lower()

    # Score each category
    category_scores: dict[str, int] = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                # Longer keyword match → higher weight
                category_scores[category] += len(kw.split())

    best_category = max(category_scores, key=lambda c: category_scores[c])
    if category_scores[best_category] == 0:
        return "general"
    return best_category


def _build_normalized_query(original: str, expansions: list[str]) -> str:
    """
    Construct the final normalized query string by appending
    expansion terms to the cleaned original.
    """
    cleaned = _clean_text(original)
    tokens = cleaned.split()
    tokens = _remove_noise_words(tokens)
    base = " ".join(tokens)

    if expansions:
        expansion_str = " ".join(expansions)
        return f"{base} {expansion_str}".strip()
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 5. PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────


def normalize_query(query: str) -> dict:
    """
    Normalize a raw user query into a structured dict ready for the reranker.

    Parameters
    ----------
    query : str
        Raw natural-language product description from the user.

    Returns
    -------
    dict with keys:
        original_query    : str  – untouched input
        normalized_query  : str  – cleaned + synonym-expanded query
        category          : str  – detected BIS category
        expansions        : list – synonym terms that were added
        tokens            : list – final token list (for BM25 use)
    """
    if not query or not query.strip():
        logger.warning("normalize_query received an empty query.")
        return {
            "original_query": query,
            "normalized_query": "",
            "category": "general",
            "expansions": [],
            "tokens": [],
        }

    original = query.strip()
    expansions = _expand_synonyms(original)
    normalized = _build_normalized_query(original, expansions)
    category = _detect_category(normalized)
    tokens = [t for t in normalized.split() if len(t) > 1]

    result = {
        "original_query": original,
        "normalized_query": normalized,
        "category": category,
        "expansions": expansions,
        "tokens": tokens,
    }

    logger.debug(
        "normalize_query | category=%s | expansions=%s | normalized='%s'",
        category,
        expansions,
        normalized,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXAMPLE USAGE  (run: python -m src.query_normalizer)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.DEBUG)

    test_queries = [
        "I need TMT bars for construction of RCC columns",
        "What standard covers OPC cement for general construction?",
        "Crushed stone aggregate for concrete mix design",
        "Ready mix concrete M30 grade for slab",
        "Fly ash bricks for load bearing walls",
        "PVC pipes for water supply in residential building",
        "Vitrified tiles for flooring",
        "Portland pozzolana cement for plastering",
    ]

    print("\n" + "═" * 70)
    print("  Query Normalizer – BIS Standards RAG System")
    print("═" * 70)

    for q in test_queries:
        result = normalize_query(q)
        print(f"\n  INPUT    : {result['original_query']}")
        print(f"  CATEGORY : {result['category']}")
        print(f"  EXPANDED : {result['normalized_query']}")
        print(f"  TOKENS   : {result['tokens'][:8]}{'...' if len(result['tokens'])>8 else ''}")
        print("  " + "─" * 66)
