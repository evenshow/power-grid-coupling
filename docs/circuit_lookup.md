# Circuit lookup build process

This document describes the preprocessing workflow implemented in [`scripts/build_circuit_lookup.py`](../scripts/build_circuit_lookup.py).

## Inputs

* `data/ltds-appendix-3.csv` – Appendix 3 circuit parameters, including the "S/S or Busbar Name Node 1/2" columns and operating voltages.
* `data/npg-ehv-feeders.csv` – geospatial feeder sections with circuit and section names, voltages, and section identifiers.

## Normalisation

The script loads both CSV files with pandas and applies the following text normalisation steps:

1. Upper-case and trim whitespace for all circuit or busbar labels.
2. Standardise dash characters and collapse repeated whitespace.
3. Expand common abbreviations such as `T1/T2`, `T1`, and `GT1` into descriptive forms (e.g. `TRANSFORMER 1`).
4. Remove high-volume asset codes (patterns such as `NCTM-00779289`) and standalone numeric identifiers that do not help with matching.

The helper utilities in the script capture this behaviour:

* `normalize_text` performs the general clean-up and abbreviation expansion.
* `scrub_fragment` strips asset codes, numeric-only tokens, and unhelpful stop-words while keeping compound names intact.
* `base_label` removes explicit transformer identifiers to expose the underlying substation or line name so that different naming schemes can be compared.

## Candidate endpoints

For each feeder record the script parses both `Circuit Section Name` and `Circuit Name`. The `extract_candidates` helper splits strings on connectors (dashes, parentheses, "TEE TO", etc.) and returns a deduplicated list of candidate endpoint phrases. A second pass (`base_label`) reduces those phrases to base names, dropping residual transformer numbers or voltage suffixes.

Only the first six distinct base labels are retained per section (longer phrases first) to avoid combinatorial explosions when exploring matches. The retained base labels are stored in the output for auditing.

## Matching logic

Matching proceeds in three stages for every feeder section:

1. **Direct base-pair lookup.** The script pre-computes an index of Appendix 3 rows keyed by the pair of base node names (ignoring order). If any base pair from the feeder record matches this index, the corresponding Appendix rows are considered candidates.
2. **Token-guided suggestions.** When no direct base pair is found, token-level lookups are used to propose likely Appendix base names. Up to three suggestions per feeder base label are combined and checked against the Appendix pair index. Any hits are folded back into the candidate list.
3. **Unmatched fall-back.** If the previous steps fail, the record is flagged as `unmatched`. The script still records any Grid Supply Point (GSP) hints discovered from token overlap to guide manual reconciliation.

For matched base pairs the algorithm applies an additional voltage consistency check (comparing feeder voltage in volts with Appendix operating voltage in kV). Rows that clear this filter uniquely are labelled `matched`; otherwise they are marked `ambiguous` with all plausible Appendix indices retained for follow-up review.

## Outputs

Two artefacts are produced under `data/derived/`:

* `circuit_lookup.csv` – the main crosswalk including section identifiers, normalised labels, candidate bases, match status, any matched Appendix indices, and notes describing how the match was obtained.
* `circuit_lookup_summary.json` – summary counts of matched, ambiguous, and unmatched sections to track coverage.

Regenerating the lookup is as simple as running:

```bash
python scripts/build_circuit_lookup.py
```

The current run matched 1,220 of 5,001 feeder sections directly and surfaced an additional 1,200 ambiguous candidates for manual confirmation (see the JSON summary for exact counts). The remaining sections are flagged as unmatched with suggested GSPs to speed targeted investigation.

## Next steps

* Review the `ambiguous` rows in `data/derived/circuit_lookup.csv`, prioritising cases where multiple Appendix entries share the same base names but different transformer identifiers.
* Use the GSP hints for `unmatched` rows to cross-reference with other datasets or internal knowledge to refine the mapping.
* Extend `base_label` or the token suggestion logic if additional naming conventions surface during manual validation.
