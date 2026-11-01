PY ?= python3
.PHONY: all data test experiments report paper clean overleaf release tabpfn review

all: data test experiments report paper

data:
	$(PY) scripts/download_tabular.py

test:
	$(PY) -m pytest -q tests

experiments:
	$(PY) experiments/run.py all 2>&1 | tee results/run_log.txt

report:
	$(PY) experiments/report.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex manuscript_anonymised.tex

clean:
	rm -rf results/*.csv results/*.npz results/*.json data/processed paper/generated paper/*.aux paper/*.log paper/*.bbl paper/*.blg paper/*.fls paper/*.fdb_latexmk paper/*.out

runbook:
	pandoc RUNBOOK.md CLAUDE.md -s -o docs/runbook.tex --pdf-engine=xelatex -V geometry:margin=2cm -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" -V fontsize=10pt -V colorlinks=true --toc -M title="Code run-book for Claude Code: bc-grade-opendata v1.0" && sed -i '/usepackage{lmodern}/d' docs/runbook.tex && cd docs && xelatex -interaction=nonstopmode runbook.tex && xelatex -interaction=nonstopmode runbook.tex

overleaf:   ## Overleaf upload: named + blinded manuscript, title page, highlights, graphical abstract
	cd paper && rm -f ../dist/overleaf.zip && mkdir -p ../dist && zip -qr ../dist/overleaf.zip \
	  *.tex refs.bib sections figures generated -x "generated/*.png"

release:    ## compact reviewer package (code, configs, tests, results CSV/JSON, paper sources; no data, no npz)
	mkdir -p dist && git archive --format=zip -o dist/bc-grade-opendata.zip HEAD

tabpfn:     ## optional TabPFN rung: export HF_TOKEN=... (licence accepted at huggingface.co/Prior-Labs) first
	$(PY) -m pip install "tabpfn>=2"
	$(PY) experiments/run.py e2opt e3 e5 e7 && $(PY) experiments/report.py && $(MAKE) paper

review:     ## anonymised reviewer archive (tracked files minus export-ignore paths)
	mkdir -p dist && git archive --format=zip --prefix=bc-grade-opendata/ -o dist/bc-grade-opendata-review.zip HEAD
