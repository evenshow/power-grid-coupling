# power-grid-coupling

## Circuit lookup preprocessing

The repository now includes a preprocessing script that aligns Northern Powergrid feeder geometries with Appendix 3 circuit parameters:

```bash
python scripts/build_circuit_lookup.py
```

Running the script regenerates `data/derived/circuit_lookup.csv` and `data/derived/circuit_lookup_summary.json`. See [docs/circuit_lookup.md](docs/circuit_lookup.md) for a detailed description of the normalisation, matching logic, and follow-up actions for ambiguous or unmatched rows.
