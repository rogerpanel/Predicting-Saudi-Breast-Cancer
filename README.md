# bc-grade-opendata

**Predicting breast cancer grade from routine clinical data: an open multi-cohort study with transport to Saudi patients**: reproducibility package
(code, configuration, result files and manuscript sources).

Anyone can re-run everything from public data: **no ethics approval, data-use agreement or account is
needed**. Every table, figure and in-text number in the manuscript is generated from `results/` by
`experiments/report.py`.

## Quick start

```bash
git clone <this repository> && cd bc-grade-opendata          # or download the ZIP and unzip it
python3 -m venv .venv && . .venv/bin/activate                 # Python 3.11+ (paper run: 3.13)
pip install -r requirements.txt
make data test          # download the open data (~10 MB), verify SHA-256 checksums, run 12 guard tests
make experiments        # E1–E10 (≈ 50 min on 4 CPU cores at the paper settings)
make report paper       # regenerate tables, figures, numbers.tex and the PDF (needs TeX Live + latexmk)
```

`make all` runs the whole chain. For a 15-minute check, set `cv.repeats: 2`, `bootstrap: 200`,
`mi.m: 2`, `transport_boot: 20` and `radiomics.repeats: 2` in `configs/experiment.yaml`
(numbers will differ slightly from the paper; the seed is fixed, so the paper settings reproduce
`paper/generated/numbers.tex` exactly).

Re-run a single stage, e.g. the leakage audit: `python experiments/run.py e4 && python experiments/report.py`.

## Data (downloaded, never redistributed)

| Cohort | Source | Country | Role | Graded n |
|---|---|---|---|---|
| METABRIC | cBioPortal `brca_metabric` | UK / Canada | development | 2386 |
| MSK-2018 | cBioPortal `breast_msk_2018` | USA | external (full features) | 1573 |
| Duke-Breast-Cancer-MRI | TCIA clinical + imaging-feature XLSX (checksummed GitHub mirror) | USA | external (full) + radiomics | 637 |
| MBCproject | cBioPortal `brca_mbcproject_2022` | USA / Canada | external (core) | 207 |
| Sanger | cBioPortal `brca_sanger` | UK / Europe | external (core) + leakage demo | 100 |
| Broad | cBioPortal `brca_broad` | Mexico, Vietnam | external (core) | 59 |
| BCCRC | cBioPortal `brca_bccrc` | Canada (TNBC) | descriptive | 63 |
| TCGA-BRCA | cBioPortal PanCancer Atlas | USA | descriptive (no grade) | – |
| Ljubljana | UCI id 14 (GitHub mirror) | Slovenia | sensitivity | 286 |

`scripts/download_tabular.py` fetches the files and checks them against `data/raw/MANIFEST.json`
(it warns if a custodian has updated a file). Duke data are CC BY-NC 4.0 (non-commercial use).
Saudi context enters only through published aggregate figures, transcribed with table and page
references in `configs/anchors.yaml`.

## What the experiments do

| Stage | Question | Key output |
|---|---|---|
| E1 | Cohort profiles; Kendall τb / Cramér's V / Spearman ρ with bootstrap CIs and BH-FDR | `tab_cohorts`, `tab_associations` |
| E2 | Nested 10×5 CV model ladder in METABRIC (baseline → LR → ordinal LR → kNN → RF → XGBoost → LightGBM); paired Δ vs LR | `tab_internal` |
| E2mi | Multiple imputation (m = 20, pooled predictions) and complete-case sensitivity | `tab_mi` |
| E3 | Frozen external validation (MSK-2018, Duke full; + MBCproject, Sanger, Broad core); leave-one-dataset-out | `tab_external`, `tab_lodo` |
| E4 | Leakage audit: NPI, SMOTE/imputation before split, grade components, record-level split | `tab_leakage` |
| E5 | Calibration, intercept update, per-grade O/E, decision curves | `tab_calibration`, `tab_oe` |
| E6 | SHAP direction tests vs pre-specified expectations; ordinal odds ratios | `tab_shap` |
| E7 | Raking to published Saudi case-mix (two anchors, bootstrap CIs, pooled anchor, age sensitivity) | `tab_transport`, `tab_agesens` |
| E8 | Ljubljana sensitivity | `tab_ljubljana` |
| E9 | Duke: clinical vs DCE-MRI radiomics vs fusion (nested CV) | `tab_duke_radiomics` |
| E10 | Riley/Pate multinomial sample size | `tab_samplesize` |

Figures are native pgfplots (`paper/figures/*_pgf.tex`) reading `paper/generated/data/*.csv`;
Figure 1 (study framework) is TikZ.

## Optional: TabPFN (tabular foundation model)

The pre-specified TabPFN rung needs gated weights from Hugging Face:
1. accept the model licence at https://huggingface.co/Prior-Labs (logged in);
2. create a read token and `export HF_TOKEN=hf_...`;
3. `make tabpfn` — adds TabPFN to E2/E3/E5/E7 without re-running the rest, then rebuilds the paper
   (the Methods/Limitations text updates automatically).

## Safeguards (`tests/`)
In-situ disease is excluded, never relabelled; ILC is never merged with IDC; one row per patient;
imputation, scaling and resampling are fitted inside training folds only; grade-encoding columns (NPI,
Nottingham component scores) are never predictors; multiple-imputation predictions are pooled over
distinct imputations.

## Layout
```
configs/      datasets.yaml · experiment.yaml (pre-specified plan, seed) · anchors.yaml (Saudi published figures)
scripts/      download_tabular.py · download_imaging.py (image-level arm, not run) · flatten_tex.py
src/bcgrade/  harmonise.py · describe.py · models.py · evaluate.py
experiments/  run.py (E1–E10) · report.py (tables/figures/macros)
tests/        pytest guards
results/      CSV/JSON results of the paper run (large .npz prediction files are regenerated, not stored)
paper/        main.tex · manuscript_anonymised.tex · sections/ · figures/ · generated/ · refs.bib
```

## Licence and citation
Code: MIT (`LICENSE`). Data remain under their custodians' terms. Citation details in `CITATION.cff`
(completed after review).
