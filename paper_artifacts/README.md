# Frozen paper artifacts

This directory contains the exact data and figures used by the revised paper.

- `tables/`: CSV inputs for all three paper figures and numerical statements;
- `figures/`: vector PDF and 600-dpi PNG exports;
- `reference_manifest.json`: frozen protocol plus SHA-256 hashes for every CSV;
- `RESULTS_SUMMARY.md`: human-readable summary generated from the frozen tables;
- `ARTIFACTS.sha256`: checksums for the complete frozen artifact directory.

Validate the table protocol and manifest with:

```bash
python main.py check-figures
```

Render the figures from the frozen tables without rerunning simulations with:

```bash
python main.py figures
```

The files are refreshed only by an explicit successful run of:

```bash
python main.py figure-pipeline --workers 4 --refresh-artifacts
```
