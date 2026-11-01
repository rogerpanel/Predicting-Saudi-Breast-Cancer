#!/usr/bin/env python3
"""Inline every \\input{...} of paper/main.tex into one .tex file (numbers, sections, tables, TikZ).
pgfplots still reads generated/data/*.csv, so ship that folder with the flattened file.
Usage: python scripts/flatten_tex.py paper/main.tex dist/manuscript_v1.0.tex"""
import re
import sys
from pathlib import Path

PAT = re.compile(r"\\input\{([^}]+)\}")


def expand(path: Path, root: Path) -> str:
    out = []
    for line in path.read_text().splitlines(keepends=True):
        if line.lstrip().startswith("%"):
            out.append(line)
            continue
        def sub(m):
            f = root / m.group(1)
            f = f if f.suffix else f.with_suffix(".tex")
            if not f.exists():            # e.g. authors.tex absent in the anonymised tree: keep the \input
                return m.group(0)
            return f"% --- begin {m.group(1)}\n" + expand(f, root).rstrip("\n") + f"\n% --- end {m.group(1)}\n"
        out.append(PAT.sub(sub, line))
    return "".join(out)


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(expand(src, src.parent))
    print(f"wrote {dst}")
