#!/usr/bin/env python3
"""Download every open tabular source into data/raw/ and record SHA-256 checksums.

No account, approval or data-use agreement is needed for any file fetched here.
Usage:  python scripts/download_tabular.py [--force]
"""
import argparse, hashlib, json, sys, time, urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FILES = ["data_clinical_patient.txt", "data_clinical_sample.txt"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
            return
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(3)
    raise RuntimeError(f"failed to download {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "datasets.yaml"))
    base = cfg["cbioportal_base"]
    manifest = {}
    for name, d in cfg["tabular"].items():
        if "study_id" in d:
            for fn in FILES:
                dest = RAW / name / fn
                fetch(f"{base}/{d['study_id']}/{fn}", dest, args.force)
                manifest[f"{name}/{fn}"] = sha256(dest)
        elif "files" in d:
            for url in d["files"]:
                dest = RAW / name / Path(url).name
                fetch(url, dest, args.force)
                manifest[f"{name}/{dest.name}"] = sha256(dest)
        else:
            dest = RAW / name / Path(d["url"]).name
            fetch(d["url"], dest, args.force)
            manifest[f"{name}/{dest.name}"] = sha256(dest)
        print(f"ok  {name}")
    out = RAW / "MANIFEST.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    changed = [k for k in manifest if k in prev and prev[k] != manifest[k]]
    if changed:
        print("WARNING: upstream files changed since last manifest:", changed)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"checksums -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
