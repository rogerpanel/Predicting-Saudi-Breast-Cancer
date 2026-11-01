#!/usr/bin/env python3
"""Arm B (imaging) downloader -- NOT executed in v0.1.

Fetches the Saudi/MENA-centred imaging sets used for the secondary endpoints:
  * KAU-BCMD (Saudi Arabia, King Abdulaziz University) -- Kaggle, free account
  * BUSI (Egypt, Baheya Hospital ultrasound)            -- Kaggle, free account
  * CDD-CESM (Egypt, NCI Cairo)                         -- TCIA, open
  * CBIS-DDSM, CMMD                                     -- TCIA, open (non-MENA comparators)

Prerequisites
  Kaggle: create ~/.kaggle/kaggle.json (Account -> Create API token); pip install kaggle
  TCIA:   pip install tcia_utils   (NBIA REST, no login for public collections)

Usage
  python scripts/download_imaging.py kau_bcmd busi            # Kaggle sets
  python scripts/download_imaging.py cdd_cesm --manifest-only  # write TCIA series list first
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw_imaging"


def kaggle(dataset_id: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(dest), "--unzip"], check=True)


def tcia(collection: str, dest: Path, manifest_only: bool):
    from tcia_utils import nbia  # type: ignore
    dest.mkdir(parents=True, exist_ok=True)
    series = nbia.getSeries(collection=collection)
    import pandas as pd
    pd.DataFrame(series).to_csv(dest / "series_manifest.csv", index=False)
    print(f"{collection}: {len(series)} series listed -> {dest/'series_manifest.csv'}")
    if not manifest_only:
        nbia.downloadSeries(series, path=str(dest), csv_filename=str(dest / "download_log"))


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "datasets.yaml"))["imaging"]
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=[k for k, v in cfg.items() if v["access"] != "dua"])
    ap.add_argument("--manifest-only", action="store_true")
    a = ap.parse_args()
    for n in a.names:
        d = cfg[n]
        if d["host"] == "kaggle":
            kaggle(d["id"], OUT / n)
        elif d["host"] == "tcia":
            tcia(d["collection"], OUT / n, a.manifest_only)
        else:
            sys.exit(f"{n}: host {d['host']} not supported")


if __name__ == "__main__":
    main()
