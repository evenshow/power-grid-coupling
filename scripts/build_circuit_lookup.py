"""Generate a crosswalk between feeder circuit sections and Appendix 3 circuit data."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

DATA_DIR = Path("data")
APPENDIX3_PATH = DATA_DIR / "ltds-appendix-3.csv"
FEEDERS_PATH = DATA_DIR / "npg-ehv-feeders.csv"
DERIVED_DIR = DATA_DIR / "derived"
OUTPUT_PATH = DERIVED_DIR / "circuit_lookup.csv"

CONNECTOR_PATTERN = re.compile(r"\s*(?:-+|–|—|\bTEE TO\b|\bTEE\b|\bTO\b|:|;|,|\(|\)|/|\\)\s*")
ASSET_CODE_PATTERN = re.compile(r"\b[A-Z]{2,5}[- ]?\d{5,}\b")
MULTISPACE = re.compile(r"\s+")
RESOLVE_CACHE: dict[Tuple[Tuple[str, ...], Optional[int]], MatchResult] = {}

STOPWORD_REPLACEMENTS = (
    (re.compile(r"\bTRANSFORMER NO\.\s*(\d+)\b"), r"TRANSFORMER \1"),
    (re.compile(r"\bN\.\s*\d+\b"), " "),
)

ABBREVIATION_PATTERNS = (
    (re.compile(r"\bT(\d+)/(\d+)\b"), lambda m: f"TRANSFORMER {m.group(1)} / TRANSFORMER {m.group(2)}"),
    (re.compile(r"\bT(\d+)\b"), lambda m: f"TRANSFORMER {m.group(1)}"),
    (re.compile(r"\bGT(\d+)/(\d+)\b"), lambda m: f"GRID TRANSFORMER {m.group(1)} / GRID TRANSFORMER {m.group(2)}"),
    (re.compile(r"\bGT(\d+)\b"), lambda m: f"GRID TRANSFORMER {m.group(1)}"),
    (re.compile(r"\bS/S\b"), "SUBSTATION"),
)

DROP_WORDS = {
    "TEE",
    "TEED",
    "TO",
    "CIRCUIT",
    "BREAKER",
    "INTERNALS",
    "SECTION",
    "ISOL",
    "ISOLATED",
    "ISOLATOR",
    "EHV",
    "NO",
    "ID",
    "ASSET",
    "NUMBER",
}


@dataclass
class AppendixNode:
    section_index: int
    node_position: int
    node_label: str
    base_label: str
    gsp: str
    operating_voltage_kv: Optional[float]


@dataclass
class MatchResult:
    status: str
    appendix_indices: List[int]
    node_labels: List[str]
    base_labels: List[str]
    gsp_candidates: List[str]
    notes: Optional[str] = None


def normalize_text(value: object) -> Optional[str]:
    """Upper-case, trim, and expand common abbreviations."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.upper()
    text = text.replace("–", "-").replace("—", "-")
    text = MULTISPACE.sub(" ", text)
    for pattern, replacement in ABBREVIATION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def scrub_fragment(fragment: str) -> str:
    fragment = ASSET_CODE_PATTERN.sub(" ", fragment)
    for pattern, replacement in STOPWORD_REPLACEMENTS:
        fragment = pattern.sub(replacement, fragment)
    words = []
    for token in fragment.split():
        if token in DROP_WORDS:
            continue
        if token.endswith("KV") and token[:-2].isdigit():
            continue
        if token.isdigit():
            continue
        words.append(token)
    cleaned = " ".join(words)
    cleaned = cleaned.replace("-", " ")
    cleaned = MULTISPACE.sub(" ", cleaned)
    return cleaned.strip(" -")


def extract_candidates(*values: object) -> List[str]:
    candidates: List[str] = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized:
            continue
        pieces = CONNECTOR_PATTERN.split(normalized)
        for piece in pieces:
            piece = piece.strip(" -")
            if not piece:
                continue
            cleaned = scrub_fragment(piece)
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
    return candidates


def base_label(label: str) -> str:
    base = re.sub(r"\bGRID TRANSFORMER\s+\d+\b", "", label)
    base = re.sub(r"\bTRANSFORMER\s+\d+\b", "", base)
    base = re.sub(r"\bGT\s+\d+\b", "", base)
    base = re.sub(r"\b\d+/\d+KV\b", "", base)
    base = re.sub(r"\b\d+KV\b", "", base)
    base = re.sub(r"\b\d+\b", "", base)
    base = base.replace("-", " ")
    base = MULTISPACE.sub(" ", base)
    return base.strip(" -")


def load_appendix_nodes() -> Tuple[pd.DataFrame, dict[str, List[AppendixNode]], dict[str, set[str]]]:
    appendix = pd.read_csv(APPENDIX3_PATH)
    appendix["Node 1 Normalized"] = appendix["S/S or Busbar Name Node 1"].map(normalize_text)
    appendix["Node 2 Normalized"] = appendix["S/S or Busbar Name Node 2"].map(normalize_text)
    appendix["Node 1 Base"] = appendix["Node 1 Normalized"].map(lambda x: base_label(x) if isinstance(x, str) else None)
    appendix["Node 2 Base"] = appendix["Node 2 Normalized"].map(lambda x: base_label(x) if isinstance(x, str) else None)

    by_base: dict[str, List[AppendixNode]] = {}
    token_index: dict[str, set[str]] = {}
    for idx, row in appendix.iterrows():
        for position in (1, 2):
            label = row[f"Node {position} Normalized"]
            if not isinstance(label, str):
                continue
            base = row[f"Node {position} Base"] or ""
            if base:
                node = AppendixNode(
                    section_index=idx,
                    node_position=position,
                    node_label=label,
                    base_label=base,
                    gsp=row.get("GSP", ""),
                    operating_voltage_kv=row.get("Operating Voltage (kV)") if not math.isnan(row.get("Operating Voltage (kV)", math.nan)) else None,
                )
                by_base.setdefault(base, []).append(node)
                for token in base.split():
                    token_index.setdefault(token, set()).add(base)
    return appendix, by_base, token_index


def build_base_pair_index(appendix: pd.DataFrame) -> dict:
    index = {}
    for idx, row in appendix.iterrows():
        base1 = row.get("Node 1 Base")
        base2 = row.get("Node 2 Base")
        if not isinstance(base1, str) or not isinstance(base2, str) or not base1 or not base2:
            continue
        key = tuple(sorted((base1, base2)))
        index.setdefault(key, []).append(idx)
    return index


def resolve_match(
    candidate_bases: Sequence[str],
    voltage: Optional[float],
    appendix: pd.DataFrame,
    base_pair_index: dict,
    appendix_nodes_by_base: dict[str, List[AppendixNode]],
    appendix_base_token_index: dict[str, set[str]],
) -> MatchResult:
    base_pairs = []
    unique_bases = tuple(sorted({base for base in candidate_bases if base}))
    voltage_key = int(voltage) if isinstance(voltage, (int, float)) and not math.isnan(voltage) else None
    cache_key = (unique_bases, voltage_key)
    if cache_key in RESOLVE_CACHE:
        cached = RESOLVE_CACHE[cache_key]
        return MatchResult(
            status=cached.status,
            appendix_indices=list(cached.appendix_indices),
            node_labels=list(cached.node_labels),
            base_labels=list(cached.base_labels),
            gsp_candidates=list(cached.gsp_candidates),
            notes=cached.notes,
        )
    derived_base_labels: set[str] = set(unique_bases)
    for pair in combinations(unique_bases, 2):
        key = tuple(sorted(pair))
        if key in base_pair_index:
            base_pairs.append((pair, base_pair_index[key]))

    if not base_pairs:
        suggestion_lists: List[List[str]] = []
        for base in unique_bases:
            suggestions: set[str] = set()
            for token in base.split():
                suggestions.update(appendix_base_token_index.get(token, set()))
            if suggestions:
                suggestion_lists.append(sorted(suggestions)[:3])
        if len(suggestion_lists) >= 2:
            for idx_a in range(len(suggestion_lists)):
                for idx_b in range(idx_a + 1, len(suggestion_lists)):
                    for label_a in suggestion_lists[idx_a]:
                        for label_b in suggestion_lists[idx_b]:
                            key = tuple(sorted((label_a, label_b)))
                            if key in base_pair_index:
                                base_pairs.append(((label_a, label_b), base_pair_index[key]))
                                derived_base_labels.update({label_a, label_b})

    if not base_pairs:
        hinted_gsps: set[str] = set()
        for base in unique_bases:
            for token in base.split():
                for hint_base in appendix_base_token_index.get(token, set()):
                    for node in appendix_nodes_by_base.get(hint_base, []):
                        if node.gsp:
                            hinted_gsps.add(node.gsp)
        result = MatchResult(
            status="unmatched",
            appendix_indices=[],
            node_labels=[],
            base_labels=list(derived_base_labels),
            gsp_candidates=sorted(hinted_gsps),
            notes="No base-pair match; review suggested GSPs",
        )
        RESOLVE_CACHE[cache_key] = result
        return result

    resolved_indices: List[int] = []

    for pair, indices in base_pairs:
        if voltage is not None:
            candidates = [i for i in indices if not math.isnan(appendix.loc[i, "Operating Voltage (kV)"]) and abs(appendix.loc[i, "Operating Voltage (kV)"] * 1000 - voltage) < 1e-3]
        else:
            candidates = indices
        if len(candidates) == 1:
            resolved_indices.extend(candidates)
            continue
        if len(candidates) > 1:
            resolved_indices.extend(candidates)
            continue
        resolved_indices.extend(indices)

    resolved_indices = sorted(set(resolved_indices))
    if not resolved_indices:
        result = MatchResult(
            status="ambiguous",
            appendix_indices=sorted({idx for _, idx_list in base_pairs for idx in idx_list}),
            node_labels=[],
            base_labels=list(derived_base_labels),
            gsp_candidates=[],
            notes="Base pairs found but filtered out",
        )
        RESOLVE_CACHE[cache_key] = result
        return result

    if len(resolved_indices) == 1:
        idx = resolved_indices[0]
        row = appendix.loc[idx]
        node_labels = [row.get("Node 1 Normalized", ""), row.get("Node 2 Normalized", "")]
        gsps = [row.get("GSP", "")] if isinstance(row.get("GSP", ""), str) else []
        result = MatchResult(status="matched", appendix_indices=[idx], node_labels=node_labels, base_labels=list(derived_base_labels), gsp_candidates=gsps, notes="Unique base pair match")
        RESOLVE_CACHE[cache_key] = result
        return result

    result = MatchResult(
        status="ambiguous",
        appendix_indices=resolved_indices,
        node_labels=[],
        base_labels=list(derived_base_labels),
        gsp_candidates=sorted({appendix.loc[idx, "GSP"] for idx in resolved_indices if isinstance(appendix.loc[idx, "GSP"], str)}),
        notes="Multiple Appendix rows share the same base pair",
    )
    RESOLVE_CACHE[cache_key] = result
    return result


def build_lookup() -> pd.DataFrame:
    appendix, appendix_nodes_by_base, appendix_base_token_index = load_appendix_nodes()
    base_pair_index = build_base_pair_index(appendix)
    feeders = pd.read_csv(FEEDERS_PATH)

    records = []
    for _, row in feeders.iterrows():
        section_id = row.get("Circuit Section ID")
        if pd.isna(section_id):
            continue
        section_name = row.get("Circuit Section Name")
        circuit_name = row.get("Circuit Name")
        candidates = extract_candidates(section_name, circuit_name)
        base_candidates = [base_label(candidate) for candidate in candidates]
        base_candidates = [base for base in base_candidates if base]
        if len(base_candidates) > 6:
            base_candidates = sorted(set(base_candidates), key=lambda value: (-len(value), value))[:6]
        else:
            base_candidates = list(dict.fromkeys(base_candidates))
        voltage = row.get("voltage")
        match = resolve_match(
            base_candidates,
            voltage,
            appendix,
            base_pair_index,
            appendix_nodes_by_base,
            appendix_base_token_index,
        )

        records.append(
            {
                "circuit_section_id": section_id,
                "circuit_id": row.get("Circuit ID"),
                "line_situation": row.get("Line situation"),
                "voltage": voltage,
                "circuit_section_name_normalized": normalize_text(section_name) or "",
                "circuit_name_normalized": normalize_text(circuit_name) or "",
                "candidate_labels": json.dumps(candidates, ensure_ascii=False),
                "candidate_bases": json.dumps(base_candidates, ensure_ascii=False),
                "match_status": match.status,
                "appendix_indices": json.dumps(match.appendix_indices),
                "appendix_node_labels": json.dumps(match.node_labels, ensure_ascii=False),
                "appendix_base_labels": json.dumps(match.base_labels, ensure_ascii=False),
                "gsp_candidates": json.dumps(match.gsp_candidates, ensure_ascii=False),
                "resolution_notes": match.notes or "",
            }
        )

    df = pd.DataFrame.from_records(records)
    df.sort_values("circuit_section_id", inplace=True)
    return df


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    lookup = build_lookup()
    lookup.to_csv(OUTPUT_PATH, index=False)
    summary_path = DERIVED_DIR / "circuit_lookup_summary.json"
    summary = {
        "total_sections": int(len(lookup)),
        "matched": int((lookup["match_status"] == "matched").sum()),
        "ambiguous": int((lookup["match_status"] == "ambiguous").sum()),
        "partial": int((lookup["match_status"] == "partial").sum()),
        "unmatched": int((lookup["match_status"] == "unmatched").sum()),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Lookup table written to {OUTPUT_PATH}")
    print(f"Summary written to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
