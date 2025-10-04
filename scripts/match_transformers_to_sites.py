#!/usr/bin/env python3
"""Match transformer records to site geometries.

This script normalises transformer and site names to build candidate
keys, performs token-based fuzzy matching (supplemented by spatial
proximity) and writes out match reports alongside a confirmed mapping
table that downstream tools can re-use.
"""

from __future__ import annotations

import csv
import math
import pathlib
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRANSFORMER_PATH = ROOT / "data" / "ltds-appendix-4.csv"
SITE_PATH = ROOT / "data" / "ltds_ehv_sites.csv"
MATCH_OUTPUT_PATH = ROOT / "data" / "transformer_site_match_scores.csv"
CONFIRMED_OUTPUT_PATH = ROOT / "data" / "transformer_site_mappings.csv"


_VOLTAGE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?\s*K[VW]\b", re.IGNORECASE)
_PUNCTUATION_PATTERN = re.compile(r"[^A-Z0-9]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _strip_bom(text: str) -> str:
    return text.lstrip("\ufeff")


def _normalise_name(value: str) -> str:
    """Normalise names by upper-casing, removing voltages and punctuation."""
    if value is None:
        return ""

    value = _strip_bom(value).upper()
    value = _VOLTAGE_PATTERN.sub(" ", value)
    value = _PUNCTUATION_PATTERN.sub(" ", value)
    value = _WHITESPACE_PATTERN.sub(" ", value).strip()
    return value


def _token_key(value: str) -> str:
    tokens = [token for token in _normalise_name(value).split() if token]
    return " ".join(sorted(tokens))


def _simple_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio() * 100


def _token_sort_ratio(a: str, b: str) -> float:
    return _simple_ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def _token_set_ratio(a: str, b: str) -> float:
    set_a = set(a.split())
    set_b = set(b.split())
    common = " ".join(sorted(set_a & set_b))
    diff_a = " ".join(sorted(set_a - set_b))
    diff_b = " ".join(sorted(set_b - set_a))

    scores = []
    if common:
        scores.append(_simple_ratio(common, (common + " " + diff_a).strip()))
        scores.append(_simple_ratio(common, (common + " " + diff_b).strip()))
    scores.append(_simple_ratio(" ".join(sorted(set_a)), " ".join(sorted(set_b))))
    return max(scores)


def fuzzy_score_from_normalised(norm_a: str, norm_b: str) -> float:
    ratios = [
        _simple_ratio(norm_a, norm_b),
        _token_sort_ratio(norm_a, norm_b),
        _token_set_ratio(norm_a, norm_b),
    ]
    return max(ratios)


def fuzzy_score(name_a: str, name_b: str) -> float:
    return fuzzy_score_from_normalised(
        _normalise_name(name_a), _normalise_name(name_b)
    )


def _parse_geo_point(value: str) -> Optional[Tuple[float, float]]:
    if not value:
        return None
    value = _strip_bom(value)
    parts = value.split(",")
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
    except ValueError:
        return None
    return lat, lon


def _haversine_distance_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    hav = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(hav))


@dataclass
class TransformerRecord:
    index: int
    s_group: str
    name_node: str
    secondary_name: str
    primary_voltage_kv: Optional[float]
    secondary_voltage_kv: Optional[float]
    rating_mva: Optional[float]
    method_of_earthing: str
    licence_area: str
    transformer_substation: str
    geo_point: Optional[Tuple[float, float]]
    candidate_names: List[str] = field(default_factory=list)
    candidate_keys: List[str] = field(default_factory=list)


@dataclass
class SiteRecord:
    asset_number: str
    site_name: str
    site_purpose: str
    geo_point: Optional[Tuple[float, float]]
    normalised_name: str = ""
    token_key: str = ""


@dataclass
class MatchResult:
    transformer: TransformerRecord
    site: SiteRecord
    candidate_source: str
    candidate_key: str
    name_score: float
    spatial_distance_km: Optional[float]
    combined_score: float


def load_transformers(path: pathlib.Path) -> List[TransformerRecord]:
    records: List[TransformerRecord] = []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            primary_voltage = _safe_float(row.get("Primary Voltage (kV)"))
            secondary_voltage = _safe_float(row.get("Secondary Voltage (kV)"))
            rating = _safe_float(row.get("Transformer Rating (MVA)"))
            geo_point = _parse_geo_point(row.get("Geo Point", ""))
            candidates: List[str] = []
            candidate_keys: List[str] = []
            for key in (
                row.get("S/S or Busbar Name Node", ""),
                row.get("Transformer Substation", ""),
            ):
                key = (key or "").strip()
                if key:
                    candidates.append(key)
                    candidate_keys.append(_token_key(key))
            record = TransformerRecord(
                index=idx,
                s_group=row.get("S/S Group", ""),
                name_node=row.get("S/S or Busbar Name Node", ""),
                secondary_name=row.get("S/S or Busbar Name Node 2", ""),
                primary_voltage_kv=primary_voltage,
                secondary_voltage_kv=secondary_voltage,
                rating_mva=rating,
                method_of_earthing=row.get("Method of Earthing", ""),
                licence_area=row.get("Licence Area", ""),
                transformer_substation=row.get("Transformer Substation", ""),
                geo_point=geo_point,
                candidate_names=candidates,
                candidate_keys=candidate_keys,
            )
            records.append(record)
    return records


def load_sites(path: pathlib.Path) -> List[SiteRecord]:
    records: List[SiteRecord] = []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            site_name = row.get("Site Name", "")
            record = SiteRecord(
                asset_number=row.get("Asset Number", ""),
                site_name=site_name,
                site_purpose=row.get("Site Purpose", ""),
                geo_point=_parse_geo_point(row.get("Geo Point", "")),
                normalised_name=_normalise_name(site_name),
                token_key=_token_key(site_name),
            )
            records.append(record)
    return records


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        value = value.strip()
    except AttributeError:
        return None
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _build_site_token_index(sites: Sequence[SiteRecord]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = defaultdict(list)
    for idx, site in enumerate(sites):
        for token in site.normalised_name.split():
            if idx not in index[token]:
                index[token].append(idx)
    return index


def match_transformers(
    transformers: Sequence[TransformerRecord],
    sites: Sequence[SiteRecord],
    spatial_bonus_threshold_km: float = 1.0,
    spatial_bonus: float = 5.0,
) -> List[MatchResult]:
    site_lookup = list(sites)
    site_token_index = _build_site_token_index(site_lookup)
    results: List[MatchResult] = []

    for transformer in transformers:
        best_result: Optional[MatchResult] = None
        for candidate, candidate_key in zip(
            transformer.candidate_names, transformer.candidate_keys
        ):
            candidate_norm = _normalise_name(candidate)
            if not candidate_norm:
                continue
            token_candidates = set()
            for token in candidate_norm.split():
                token_candidates.update(site_token_index.get(token, []))
            if token_candidates:
                candidate_sites = [site_lookup[idx] for idx in token_candidates]
            else:
                candidate_sites = site_lookup
            for site in candidate_sites:
                name_score = fuzzy_score_from_normalised(
                    candidate_norm, site.normalised_name
                )
                distance: Optional[float] = None
                bonus = 0.0
                if transformer.geo_point and site.geo_point:
                    distance = _haversine_distance_km(transformer.geo_point, site.geo_point)
                    if distance <= spatial_bonus_threshold_km:
                        bonus = spatial_bonus * (1 - distance / spatial_bonus_threshold_km)
                combined = name_score + bonus
                match = MatchResult(
                    transformer=transformer,
                    site=site,
                    candidate_source=candidate,
                    candidate_key=candidate_key,
                    name_score=name_score,
                    spatial_distance_km=distance,
                    combined_score=combined,
                )
                if best_result is None or combined > best_result.combined_score:
                    best_result = match
        if best_result:
            results.append(best_result)
    return results


def write_match_report(matches: Sequence[MatchResult], path: pathlib.Path) -> None:
    fieldnames = [
        "transformer_index",
        "transformer_group",
        "candidate_name",
        "candidate_key",
        "transformer_substation",
        "primary_voltage_kv",
        "secondary_voltage_kv",
        "rating_mva",
        "site_asset_number",
        "site_name",
        "site_key",
        "site_purpose",
        "name_score",
        "spatial_distance_km",
        "combined_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for match in matches:
            writer.writerow(
                {
                    "transformer_index": match.transformer.index,
                    "transformer_group": match.transformer.s_group,
                    "candidate_name": match.candidate_source,
                    "candidate_key": match.candidate_key,
                    "transformer_substation": match.transformer.transformer_substation,
                    "primary_voltage_kv": match.transformer.primary_voltage_kv,
                    "secondary_voltage_kv": match.transformer.secondary_voltage_kv,
                    "rating_mva": match.transformer.rating_mva,
                    "site_asset_number": match.site.asset_number,
                    "site_name": match.site.site_name,
                    "site_key": match.site.token_key,
                    "site_purpose": match.site.site_purpose,
                    "name_score": f"{match.name_score:.2f}",
                    "spatial_distance_km": (
                        f"{match.spatial_distance_km:.3f}" if match.spatial_distance_km is not None else ""
                    ),
                    "combined_score": f"{match.combined_score:.2f}",
                }
            )


def confirm_matches(
    matches: Sequence[MatchResult],
    manual_overrides: Optional[Dict[int, str]] = None,
    high_confidence_threshold: float = 90.0,
    medium_confidence_threshold: float = 80.0,
    max_distance_km: float = 1.0,
) -> List[MatchResult]:
    manual_overrides = manual_overrides or {}
    confirmed: List[MatchResult] = []
    for match in matches:
        override_asset = manual_overrides.get(match.transformer.index)
        if override_asset:
            if match.site.asset_number == override_asset:
                confirmed.append(match)
            else:
                continue
            continue
        if match.combined_score >= high_confidence_threshold:
            confirmed.append(match)
        elif (
            match.combined_score >= medium_confidence_threshold
            and match.spatial_distance_km is not None
            and match.spatial_distance_km <= max_distance_km
        ):
            confirmed.append(match)
    return confirmed


def write_confirmed_matches(matches: Sequence[MatchResult], path: pathlib.Path) -> None:
    fieldnames = [
        "transformer_index",
        "transformer_group",
        "candidate_name",
        "candidate_key",
        "transformer_substation",
        "primary_voltage_kv",
        "secondary_voltage_kv",
        "rating_mva",
        "site_asset_number",
        "site_name",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for match in matches:
            writer.writerow(
                {
                    "transformer_index": match.transformer.index,
                    "transformer_group": match.transformer.s_group,
                    "candidate_name": match.candidate_source,
                    "candidate_key": match.candidate_key,
                    "transformer_substation": match.transformer.transformer_substation,
                    "primary_voltage_kv": match.transformer.primary_voltage_kv,
                    "secondary_voltage_kv": match.transformer.secondary_voltage_kv,
                    "rating_mva": match.transformer.rating_mva,
                    "site_asset_number": match.site.asset_number,
                    "site_name": match.site.site_name,
                }
            )


def main() -> None:
    transformers = load_transformers(TRANSFORMER_PATH)
    sites = load_sites(SITE_PATH)
    matches = match_transformers(transformers, sites)

    write_match_report(matches, MATCH_OUTPUT_PATH)

    manual_overrides = {
        # Low-confidence name-only matches reviewed manually:
        46: "SITE-00266684",  # West Docks ↔ WEST DOCK 3638
        466: "SITE-00266801",  # Osgodby 1/2 ↔ OSGODBY 7443
        467: "SITE-00266801",  # Osgodby 1/2 ↔ OSGODBY 7443
        791: "SITE-00267168",  # Doncaster B/Thorpe Marsh GT2 ↔ DONCASTER B 45039
        792: "SITE-00267168",  # Doncaster B/Thorpe Marsh GT1 ↔ DONCASTER B 45039
        841: "SITE-00266684",  # West Docks ↔ WEST DOCK 3638
        873: "SITE-00266801",  # Osgodby 3/4 ↔ OSGODBY 7443
    }

    confirmed = confirm_matches(matches, manual_overrides=manual_overrides)
    write_confirmed_matches(confirmed, CONFIRMED_OUTPUT_PATH)

    confidence_values = [match.combined_score for match in matches]
    if confidence_values:
        print(
            "Generated match report with", len(matches), "candidates."
        )
        print(
            "High confidence matches saved:",
            len(confirmed),
            "(median score =",
            f"{statistics.median(confidence_values):.1f})",
        )
    else:
        print("No matches generated.")


if __name__ == "__main__":
    main()
