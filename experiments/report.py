#!/usr/bin/env python3
"""Turn results/ into LaTeX tables, figures and number macros under paper/generated/.
Every number quoted in the manuscript comes from here, so re-running the pipeline updates the paper."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
import json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from bcgrade.models import LABELS  # noqa: E402

# palette (validated reference instance; see dataviz skill palette.md)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#b9d3f2", "#5d98de", "#174f94"]  # grade I -> III (one hue, light -> dark)
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42, "figure.dpi": 200})

MACROS: dict[str, str] = {}


def mac(name, val, fmt="{:.2f}"):
    MACROS[name] = fmt.format(val) if not isinstance(val, str) else val


def pct(x):
    return "--" if pd.isna(x) else f"{100 * x:.1f}"


def ci(m, lo, hi, d=3):
    return f"{m:.{d}f} ({lo:.{d}f}--{hi:.{d}f})"


def save_tab(name, body):
    (OUT / f"{name}.tex").write_text(body)


PGF = OUT / "data"
PGF.mkdir(exist_ok=True)


def pgf(name, df):
    """Data for the native pgfplots figures in paper/figures/ (comma-separated, no index)."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(";", ",")
    df.to_csv(PGF / f"{name}.csv", index=False, sep=";", float_format="%.5g", na_rep="nan")


def savefig(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


anchors = yaml.safe_load(open(ROOT / "configs" / "anchors.yaml"))
SA = anchors["saudi_kussaibi2021"]
SB = anchors["saudi_durrani2025"]

# ------------------------------------------------------------------ Table 1: cohorts vs Saudi anchor
coh = pd.read_csv(RES / "e1_cohorts.csv")
flow = pd.read_csv(RES / "e1_flow.csv")
order = ["METABRIC", "MSK-2018", "Duke MRI", "MBCproject", "Sanger", "Broad (MX/VN)", "BCCRC (TNBC)", "TCGA-BRCA"]
coh = coh.set_index("cohort").loc[order]
cols = list(coh.index) + ["Saudi (KFUH)", "Saudi (KFMC)"]
sa = {"n": SA["n"], "age_median": SA["median_age"], "er": SA["er_pos"], "pr": SA["pr_pos"], "her2": SA["her2_pos"],
      "g1": SA["grade"][1], "g2": SA["grade"][2], "g3": SA["grade"][3], "idc": SA["histology"]["idc"],
      "ilc": SA["histology"]["ilc"], "other": SA["histology"]["other"]}
sb = {"n": sum(SB["grade_counts"].values()), "g1": SB["grade"][1], "g2": SB["grade"][2], "g3": SB["grade"][3],
      "er": SB["er_pos"], "pr": SB["pr_pos"], "her2": SB["her2_pos"], "le50": SB["age_le50"]}
rows = [("Patients analysed, $n$", lambda c: f"{int(coh.loc[c, 'n'])}", f"{sa['n']}"),
        ("Age, median (IQR), y", lambda c: f"{coh.loc[c, 'age_median']:.0f} ({coh.loc[c, 'age_q1']:.0f}--{coh.loc[c, 'age_q3']:.0f})",
         f"{sa['age_median']}"),
        ("Age $\\le$ 50 y, \\%", lambda c: pct(coh.loc[c, "age_le50"]), "--"),
        ("Grade 1, \\%", lambda c: pct(coh.loc[c, "g1"]), pct(sa["g1"])),
        ("Grade 2, \\%", lambda c: pct(coh.loc[c, "g2"]), pct(sa["g2"])),
        ("Grade 3, \\%", lambda c: pct(coh.loc[c, "g3"]), pct(sa["g3"])),
        ("IDC / NST, \\%", lambda c: pct(coh.loc[c, "idc"]), pct(sa["idc"])),
        ("ILC, \\%", lambda c: pct(coh.loc[c, "ilc"]), pct(sa["ilc"])),
        ("Mixed ductal--lobular, \\%", lambda c: pct(coh.loc[c, "mixed"]), "--"),
        ("Other special type, \\%", lambda c: pct(coh.loc[c, "other"]), pct(sa["other"])),
        ("ER positive, \\%", lambda c: pct(coh.loc[c, "er"]), pct(sa["er"])),
        ("PR positive, \\%", lambda c: pct(coh.loc[c, "pr"]), pct(sa["pr"])),
        ("HER2 positive, \\%", lambda c: pct(coh.loc[c, "her2"]), pct(sa["her2"])),
        ("T1 ($\\le$20 mm), \\%", lambda c: pct(coh.loc[c, "t1"]), "--"),
        ("T2 (21--50 mm), \\%", lambda c: pct(coh.loc[c, "t2"]), "--"),
        ("$\\ge$T3, \\%", lambda c: pct(coh.loc[c, "t3"]), "--"),
        ("Node positive, \\%", lambda c: pct(coh.loc[c, "n_pos"]), "--"),
        ("Post-menopausal, \\%", lambda c: pct(coh.loc[c, "postmeno"]), "--")]
short = {"METABRIC": "METABRIC", "MSK-2018": "MSK", "Duke MRI": "Duke", "MBCproject": "MBCP", "Sanger": "Sanger",
         "Broad (MX/VN)": "Broad", "BCCRC (TNBC)": "BCCRC", "TCGA-BRCA": "TCGA", "Saudi (KFUH)": "Dammam$^\\dagger$",
         "Saudi (KFMC)": "Riyadh$^\\ddagger$"}
body = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
        " & " + " & ".join(short[c] for c in cols) + " \\\\",
        "Role & dev & ext & ext & ext & ext & ext & desc & desc & anchor & anchor \\\\", "\\midrule"]
sbcol = {"Patients analysed, $n$": f"{sb['n']}", "Age $\\le$ 50 y, \\%": pct(sb["le50"]), "Grade 1, \\%": pct(sb["g1"]),
         "Grade 2, \\%": pct(sb["g2"]), "Grade 3, \\%": pct(sb["g3"]), "ER positive, \\%": pct(sb["er"]),
         "PR positive, \\%": pct(sb["pr"]), "HER2 positive, \\%": pct(sb["her2"])}
for lab, f, s in rows:
    body.append(lab + " & " + " & ".join(f(c) for c in order) + f" & {s} & {sbcol.get(lab, '--')} \\\\")
body += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_cohorts", "\n".join(body))
for c, k in [("METABRIC", "Metabric"), ("MSK-2018", "Msk"), ("MBCproject", "Mbc"), ("Sanger", "Sanger"),
             ("Broad (MX/VN)", "Broad"), ("BCCRC (TNBC)", "Bccrc"), ("TCGA-BRCA", "Tcga"), ("Duke MRI", "Duke")]:
    mac(f"N{k}", int(coh.loc[c, "n"]), "{:d}")
    mac(f"GThree{k}", 100 * coh.loc[c, "g3"] if not pd.isna(coh.loc[c, "g3"]) else 0, "{:.1f}")
    mac(f"Age{k}", coh.loc[c, "age_median"], "{:.0f}")
graded = [c for c in order if not pd.isna(coh.loc[c, "g3"])]
mac("NGradedTotal", int(coh.loc[graded, "n"].sum()), "{:d}")
mac("NGradedCohorts", len(graded), "{:d}")
mac("NCohorts", len(order), "{:d}")
mac("NAllTotal", int(coh.loc[order, "n"].sum()), "{:d}")
mac("NMetabricRecords", int(flow.query("cohort=='METABRIC' and step=='records'").n.iloc[0]), "{:d}")
mac("NMskSamples", int(flow.query("cohort=='MSK-2018' and step=='samples'").n.iloc[0]), "{:d}")
mac("NMskPatients", int(flow.query("cohort=='MSK-2018' and step=='patients'").n.iloc[0]), "{:d}")

# flow table
fl = ["\\begin{tabular}{llr}", "\\toprule", "Cohort & Step & $n$ \\\\", "\\midrule"]
flow["_o"] = flow.cohort.map({c: i for i, c in enumerate(dict.fromkeys(flow.cohort))})
flow = flow.sort_values("_o", kind="stable")
for _, r in flow.iterrows():
    fl.append(f"{r.cohort} & {r.step} & {r.n} \\\\")
fl += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_flow", "\n".join(fl))

# ------------------------------------------------------------------ Figure 1: grade distributions
fig, ax = plt.subplots(figsize=(6.2, 2.6))
labs = graded + ["Saudi (Dammam)$^\\dagger$", "Saudi (Riyadh)$^\\ddagger$"]
vals = [coh.loc[c, ["g1", "g2", "g3"]].to_numpy(float) for c in graded] + [np.array([sa["g1"], sa["g2"], sa["g3"]]),
                                                                          np.array([sb["g1"], sb["g2"], sb["g3"]])]
pgf("grade_distribution", pd.DataFrame({"label": [l.replace("$^\\dagger$", "$^\\dagger$") for l in labs],
                                        "y": np.arange(len(labs))[::-1],
                                        "g1": [100 * v[0] for v in vals], "g2": [100 * v[1] for v in vals],
                                        "g3": [100 * v[2] for v in vals]}))
ypos = np.arange(len(labs))[::-1]
for yy, v in zip(ypos, vals):
    left = 0
    for k in range(3):
        ax.barh(yy, v[k], left=left, color=SEQ[k], height=0.62, edgecolor="white", linewidth=1.5)
        if v[k] > 0.07:
            ax.text(left + v[k] / 2, yy, f"{100*v[k]:.0f}", ha="center", va="center",
                    color=INK if k == 0 else "white", fontsize=7)
        left += v[k]
ax.set_yticks(ypos, labs)
ax.set_xlim(0, 1)
ax.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
ax.grid(axis="y", visible=False)
for k, l in enumerate(["Grade 1", "Grade 2", "Grade 3"]):
    ax.bar(0, 0, color=SEQ[k], label=l)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
ax.axhline(1.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
savefig(fig, "fig_grade_distribution")

# ------------------------------------------------------------------ Table 2 + Figure 2: associations
at = pd.read_csv(RES / "e1_associations.csv")
nice = {"grade": "Grade", "er": "ER+", "pr": "PR+", "her2": "HER2+", "t_cat": "T category", "n_pos": "Node+",
        "postmeno": "Post-menopausal", "age": "Age", "histo": "Histology"}
mname = {"tau_b": "$\\tau_b$", "V": "$V$", "rho": "$\\rho$"}
tb = ["\\begin{tabular}{llrrrr}", "\\toprule",
      "Pair & Measure & \\multicolumn{2}{c}{METABRIC} & \\multicolumn{2}{c}{MSK-2018} \\\\",
      "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}", " & & Effect (95\\% CI) & $p_\\mathrm{BH}$ & Effect (95\\% CI) & $p_\\mathrm{BH}$ \\\\",
      "\\midrule"]
for (a, b, k), g in at.groupby(["var_a", "var_b", "measure"], sort=False):
    cells = []
    for c in ["METABRIC", "MSK-2018"]:
        r = g[g.cohort == c]
        if len(r):
            r = r.iloc[0]
            p = "$<$0.001" if r.p_bh < 0.001 else f"{r.p_bh:.3f}"
            extra = f"; $\\varepsilon^2$={r.kw_eps2:.3f}" if k == "rho" else ""
            cells += [f"{r.estimate:+.3f} ({r.ci_lo:+.3f}, {r.ci_hi:+.3f}){extra}", p]
        else:
            cells += ["--", "--"]
    tb.append(f"{nice[a]} $\\times$ {nice[b]} & {mname[k]} & " + " & ".join(cells) + " \\\\")
tb += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_associations", "\n".join(tb))
g = at[at.cohort == "METABRIC"].set_index(["var_a", "var_b"])
mac("TauER", g.loc[("grade", "er"), "estimate"], "{:+.2f}")
mac("TauPR", g.loc[("grade", "pr"), "estimate"], "{:+.2f}")
mac("TauHER", g.loc[("grade", "her2"), "estimate"], "{:+.2f}")
mac("TauT", g.loc[("grade", "t_cat"), "estimate"], "{:+.2f}")
mac("TauN", g.loc[("grade", "n_pos"), "estimate"], "{:+.2f}")
mac("VHist", g.loc[("grade", "histo"), "estimate"], "{:.2f}")
mac("RhoAge", g.loc[("age", "grade"), "estimate"], "{:+.2f}")
mac("NTests", len(at), "{:d}")
mac("NSigTests", int((at.p_bh < 0.05).sum()), "{:d}")

fig, ax = plt.subplots(figsize=(6.2, 3.0))
grade_pairs = at[at.var_a.eq("grade") | at.var_b.eq("grade")].copy()
grade_pairs["lab"] = grade_pairs.apply(lambda r: f"{nice[r.var_a]} × {nice[r.var_b]} ({mname[r.measure].strip('$').replace(chr(92),'')})", axis=1)
labels_ = list(dict.fromkeys(grade_pairs.lab))
for j, (c, col, off) in enumerate([("METABRIC", C1, 0.15), ("MSK-2018", C2, -0.15)]):
    s = grade_pairs[grade_pairs.cohort == c].set_index("lab").loc[labels_]
    yy = np.arange(len(labels_))[::-1] + off
    ax.errorbar(s.estimate, yy, xerr=[s.estimate - s.ci_lo, s.ci_hi - s.estimate], fmt="o", ms=4, color=col,
                elinewidth=1.4, capsize=0, label=c)
ax.axvline(0, color=INK2, lw=0.8)
_pa = {"label": [l.replace("×", "$\\times$").replace("tau_b", "$\\tau_b$").replace("(rho)", "($\\rho$)").replace("(V)", "($V$)") for l in labels_],
       "y": np.arange(len(labels_))[::-1]}
for c, k in [("METABRIC", "met"), ("MSK-2018", "msk")]:
    s_ = grade_pairs[grade_pairs.cohort == c].set_index("lab").reindex(labels_)
    _pa.update({k: s_.estimate.to_numpy(), f"{k}_em": (s_.estimate - s_.ci_lo).to_numpy(), f"{k}_ep": (s_.ci_hi - s_.estimate).to_numpy()})
pgf("associations", pd.DataFrame(_pa))
ax.set_yticks(np.arange(len(labels_))[::-1], labels_)
ax.set_xlabel("Effect size (bootstrap 95% CI)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
savefig(fig, "fig_associations")

# ------------------------------------------------------------------ Table 3: model ladder internal
e2 = pd.read_csv(RES / "e2_summary.csv").set_index("model")
ladder = [m for m in ["majority", "mlr", "ordinal", "knn", "rf", "xgb", "lgbm", "tabpfn"] if m in e2.index]


def dcell(r, k="macro_auc"):
    if k == "macro_auc" and pd.isna(r.get(f"d_{k}", np.nan)):
        return "ref."
    return f"{r[f'd_{k}']:+.3f} ({r[f'd_{k}_lo']:+.3f}, {r[f'd_{k}_hi']:+.3f})"


t3 = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
      "Model & Macro-AUC (95\\% CI) & $\\Delta$ vs LR (95\\% CI) & AUC G1 & AUC G2 & AUC G3 & Macro-F1 (95\\% CI) & QWK & Brier \\\\", "\\midrule"]
for m in ladder:
    r = e2.loc[m]
    t3.append(f"{LABELS[m]} & {ci(r.macro_auc, r.macro_auc_lo, r.macro_auc_hi)} & {dcell(r) if m != 'mlr' else 'ref.'} & {r.auc_g1:.3f} & {r.auc_g2:.3f} & "
              f"{r.auc_g3:.3f} & {ci(r.macro_f1, r.macro_f1_lo, r.macro_f1_hi)} & {r.qwk:.3f} & {r.brier:.3f} \\\\")
t3 += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_internal", "\n".join(t3))
best_int = e2.drop("majority").macro_auc.idxmax()
mac("BestIntModel", LABELS[best_int], "{}")
mac("BestIntAuc", e2.loc[best_int, "macro_auc"], "{:.3f}")
mac("BestIntAucLo", e2.loc[best_int, "macro_auc_lo"], "{:.3f}")
mac("BestIntAucHi", e2.loc[best_int, "macro_auc_hi"], "{:.3f}")
mac("MlrIntAuc", e2.loc["mlr", "macro_auc"], "{:.3f}")
mac("OrdIntAuc", e2.loc["ordinal", "macro_auc"], "{:.3f}")
mac("XgbIntAuc", e2.loc["xgb", "macro_auc"], "{:.3f}")
mac("KnnIntAuc", e2.loc["knn", "macro_auc"], "{:.3f}")
mac("IntAucSpread", e2.drop("majority").macro_auc.max() - e2.drop("majority").macro_auc.min(), "{:.3f}")
mac("BestIntGThreeAuc", e2.loc[best_int, "auc_g3"], "{:.3f}")
mac("IntAucMinR", e2.drop("majority").macro_auc.min(), "{:.2f}")
mac("IntAucMaxR", e2.drop("majority").macro_auc.max(), "{:.2f}")
mac("MlrIntFOne", e2.loc["mlr", "macro_f1"], "{:.3f}")
mac("MajIntFOne", e2.loc["majority", "macro_f1"], "{:.3f}")
mac("MaxRepSd", e2.drop("majority").auc_rep_sd.max(), "{:.3f}")
nb = e2.drop(["majority", "mlr"])
mac("MaxDeltaInt", nb.d_macro_auc.abs().max(), "{:.3f}")
mac("MaxDeltaIntHi", nb.d_macro_auc_hi.max(), "{:.3f}")
mac("XgbDeltaInt", e2.loc["xgb", "d_macro_auc"], "{:+.3f}")
mac("XgbDeltaIntLo", e2.loc["xgb", "d_macro_auc_lo"], "{:+.3f}")
mac("XgbDeltaIntHi", e2.loc["xgb", "d_macro_auc_hi"], "{:+.3f}")
mac("NSigBetterThanLr", int((nb.d_macro_auc_lo > 0).sum()), "{:d}")
mac("KnnDeltaInt", e2.loc["knn", "d_macro_auc"], "{:+.3f}")
import json as _json  # noqa: E402
_st = _json.load(open(RES / "optional_models_status.json")) if (RES / "optional_models_status.json").exists() else {}
mac("TabpfnStatus", "run" if "tabpfn" in e2.index else "not run", "{}")
mac("TabpfnLimitation", "" if "tabpfn" in e2.index else
    "The pre-specified tabular foundation model (TabPFN) could not be run because its weights are distributed "
    "through a gated repository that was not reachable from the build environment; the code path is in place "
    "(\\texttt{make tabpfn}). ", "{}")

# ------------------------------------------------------------------ Table 4 + Figure 3: external & LODO
e3 = pd.read_csv(RES / "e3_external.csv")
ext = e3[e3.design == "METABRIC->ext"]
t4 = ["\\begin{tabular}{llrrrrr}", "\\toprule",
      "Target (features) & Model & $n$ & Macro-AUC (95\\% CI) & G3-AUC (95\\% CI) & Macro-F1 & Cal. slope (G3) \\\\",
      "\\midrule"]
for (fs, tgt), g in ext.groupby(["features", "target"], sort=False):
    for m in [x for x in ["majority", "mlr", "ordinal", "rf", "xgb", "lgbm", "tabpfn"] if x in set(g.model)]:
        r = g[g.model == m].iloc[0]
        t4.append(f"{tgt} ({fs}) & {LABELS[m]} & {r.n} & {ci(r.macro_auc, r.macro_auc_lo, r.macro_auc_hi)} & "
                  f"{ci(r.g3_auc, r.g3_auc_lo, r.g3_auc_hi)} & {r.macro_f1:.3f} & {'--' if pd.isna(r.g3_cal_slope) or m == 'majority' else f'{r.g3_cal_slope:.2f}'} \\\\")
    t4.append("\\addlinespace")
t4 = t4[:-1] + ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_external", "\n".join(t4))
lodo = e3[e3.design == "LODO"]
t5 = ["\\begin{tabular}{llrrrr}", "\\toprule", "Held-out cohort & Model & $n$ & Macro-AUC (95\\% CI) & G3-AUC (95\\% CI) & Macro-F1 \\\\",
      "\\midrule"]
for tgt, g in lodo.groupby("target", sort=False):
    for _, r in g.iterrows():
        t5.append(f"{tgt} & {LABELS[r.model]} & {r.n} & {ci(r.macro_auc, r.macro_auc_lo, r.macro_auc_hi)} & "
                  f"{ci(r.g3_auc, r.g3_auc_lo, r.g3_auc_hi)} & {r.macro_f1:.3f} \\\\")
    t5.append("\\addlinespace")
t5 = t5[:-1] + ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_lodo", "\n".join(t5))

mskf = ext[(ext.target == "MSK-2018") & (ext.features == "full")].set_index("model")
best_ext = mskf.drop("majority").macro_auc.idxmax()
mac("BestExtModel", LABELS[best_ext], "{}")
mac("BestExtAuc", mskf.loc[best_ext, "macro_auc"], "{:.3f}")
mac("BestExtAucLo", mskf.loc[best_ext, "macro_auc_lo"], "{:.3f}")
mac("BestExtAucHi", mskf.loc[best_ext, "macro_auc_hi"], "{:.3f}")
mac("MlrExtAuc", mskf.loc["mlr", "macro_auc"], "{:.3f}")
mac("XgbExtAuc", mskf.loc["xgb", "macro_auc"], "{:.3f}")
mac("MlrExtGThree", mskf.loc["mlr", "g3_auc"], "{:.3f}")
mac("NMskExt", int(mskf.loc["mlr", "n"]), "{:d}")
core = ext[ext.features == "core"]
mac("CoreExtAucMin", core[core.model != "majority"].macro_auc.min(), "{:.3f}")
mac("CoreExtAucMax", core[core.model != "majority"].macro_auc.max(), "{:.3f}")
mac("LodoAucMin", lodo.macro_auc.min(), "{:.3f}")
allext = pd.concat([ext[ext.model != "majority"], lodo])
mac("ExtAucMin", allext.macro_auc.min(), "{:.2f}")
mac("ExtAucMax", allext.macro_auc.max(), "{:.2f}")
coremsk = core[(core.target == "MSK-2018") & (core.model != "majority")]
mac("CoreMskMin", coremsk.macro_auc.min(), "{:.2f}")
mac("CoreMskMax", coremsk.macro_auc.max(), "{:.2f}")
fullmsk = ext[(ext.features == "full") & (ext.target == "MSK-2018") & (ext.model != "majority")]
mac("FullMskMin", fullmsk.macro_auc.min(), "{:.2f}")
mac("FullMskMax", fullmsk.macro_auc.max(), "{:.2f}")
mac("LodoAucMax", lodo.macro_auc.max(), "{:.3f}")
dkf = ext[(ext.target == "Duke MRI") & (ext.features == "full")].set_index("model")
mac("NDukeExt", int(dkf.loc["mlr", "n"]), "{:d}")
mac("MlrDukeAuc", dkf.loc["mlr", "macro_auc"], "{:.3f}")
mac("MlrDukeAucLo", dkf.loc["mlr", "macro_auc_lo"], "{:.3f}")
mac("MlrDukeAucHi", dkf.loc["mlr", "macro_auc_hi"], "{:.3f}")
mac("MlrDukeGThree", dkf.loc["mlr", "g3_auc"], "{:.3f}")
mac("FullDukeMin", dkf.drop("majority").macro_auc.min(), "{:.2f}")
mac("FullDukeMax", dkf.drop("majority").macro_auc.max(), "{:.2f}")
mac("MlrMskAucLo", mskf.loc["mlr", "macro_auc_lo"], "{:.3f}")
mac("MlrMskAucHi", mskf.loc["mlr", "macro_auc_hi"], "{:.3f}")
exd = ext[ext.model.isin(["ordinal", "rf", "xgb", "lgbm", "tabpfn"])]
mac("NExtComparisons", len(exd), "{:d}")
mac("NExtSigBetter", int((exd.d_macro_auc_lo > 0).sum()), "{:d}")
mac("NExtSigWorse", int((exd.d_macro_auc_hi < 0).sum()), "{:d}")

fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=True)
models_ = ["mlr", "ordinal", "knn", "rf", "xgb", "lgbm"] + (["tabpfn"] if "tabpfn" in e2.index else [])
yy = np.arange(len(models_))[::-1]
panels = [("Internal (METABRIC CV)", e2.loc[models_, ["macro_auc", "macro_auc_lo", "macro_auc_hi"]]),
          ("External: MSK-2018 (full)", ext[(ext.target == "MSK-2018") & (ext.features == "full")].set_index("model").reindex(models_)[["macro_auc", "macro_auc_lo", "macro_auc_hi"]]),
          ("External: Duke MRI (full)", ext[(ext.target == "Duke MRI") & (ext.features == "full")].set_index("model").reindex(models_)[["macro_auc", "macro_auc_lo", "macro_auc_hi"]]),
          ("External: MBCproject (core)", ext[(ext.target == "MBCproject") & (ext.features == "core")].set_index("model").reindex(models_)[["macro_auc", "macro_auc_lo", "macro_auc_hi"]])]
_pl = {"label": [LABELS[m] for m in models_], "y": yy}
for (t, s_), k in zip(panels, ["int", "msk", "duke", "mbcp"]):
    _pl.update({k: s_.macro_auc.to_numpy(), f"{k}_em": (s_.macro_auc - s_.macro_auc_lo).to_numpy(),
                f"{k}_ep": (s_.macro_auc_hi - s_.macro_auc).to_numpy()})
pgf("model_ladder", pd.DataFrame(_pl))
for ax, (t, s), col in zip(axes, panels, [C1, C2, C4, C3]):
    ok = s.macro_auc.notna().to_numpy()
    ax.errorbar(s.macro_auc[ok], yy[ok], xerr=[(s.macro_auc - s.macro_auc_lo)[ok], (s.macro_auc_hi - s.macro_auc)[ok]],
                fmt="o", ms=4, color=col, elinewidth=1.4)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (2, 2)))
    ax.set_title(t, fontsize=7.5, color=INK)
    ax.set_xlim(0.45, 0.85)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(yy, [LABELS[m] for m in models_])
fig.supxlabel("Macro-averaged one-vs-rest AUC (95% CI); dashed = chance", fontsize=8)
savefig(fig, "fig_model_ladder")

# ------------------------------------------------------------------ Table 6 + Figure 4: leakage
lk = pd.read_csv(RES / "e4_leakage.csv")
t6 = ["\\begin{tabular}{p{6.2cm}lrrr}", "\\toprule", "Scenario & Model & Honest AUC & Leaky AUC & Inflation \\\\", "\\midrule"]
for _, r in lk.iterrows():
    t6.append(f"{r.scenario} & {LABELS[r.model]} & {r.honest_auc:.3f} & {r.leaky_auc:.3f} & {r.leaky_auc - r.honest_auc:+.3f} \\\\")
t6 += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_leakage", "\n".join(t6))
mac("LeakMaxInfl", (lk.leaky_auc - lk.honest_auc).max(), "{:+.3f}")
npi = lk[lk.scenario.str.contains("NPI")].set_index("model")
mac("NpiHonest", npi.loc["xgb", "honest_auc"], "{:.3f}")
mac("NpiLeaky", npi.loc["xgb", "leaky_auc"], "{:.3f}")
smo = lk[lk.scenario.str.contains("SMOTE")].set_index("model")
mac("SmoteHonestKnn", smo.loc["knn", "honest_auc"], "{:.3f}")
mac("SmoteLeakyKnn", smo.loc["knn", "leaky_auc"], "{:.3f}")
sg = lk[lk.scenario.str.contains("components") & lk.scenario.str.contains("Sanger")].iloc[0]
sd = lk[lk.scenario.str.contains("components") & lk.scenario.str.contains("Duke")].iloc[0]
mac("CompHonestDuke", sd.honest_auc, "{:.3f}")
mac("CompLeakyDuke", sd.leaky_auc, "{:.3f}")
mac("CompHonest", sg.honest_auc, "{:.3f}")
mac("CompLeaky", sg.leaky_auc, "{:.3f}")
rec = lk[lk.scenario.str.contains("Record")].set_index("model")
mac("RecHonestKnn", rec.loc["knn", "honest_auc"], "{:.3f}")
mac("RecLeakyKnn", rec.loc["knn", "leaky_auc"], "{:.3f}")

fig, ax = plt.subplots(figsize=(6.2, 2.4))
short_s = {"Target": "NPI as predictor", "SMOTE": "SMOTE/impute before split", "Grade components": "Grade components",
           "Record": "Record-level split"}
lk["lab"] = lk.scenario.map(lambda x: next(v for k, v in short_s.items() if x.startswith(k))) + " · " + lk.model.map(LABELS)
yy = np.arange(len(lk))[::-1]
pgf("leakage", pd.DataFrame({"label": lk.lab.str.replace("·", "--").to_numpy(), "y": yy, "honest": lk.honest_auc,
                             "leaky": lk.leaky_auc}))
for y_, (_, r) in zip(yy, lk.iterrows()):
    ax.plot([r.honest_auc, r.leaky_auc], [y_, y_], color=GRID, lw=2.2, zorder=1)
ax.scatter(lk.honest_auc, yy, color=C1, s=22, zorder=2, label="Correct protocol")
ax.scatter(lk.leaky_auc, yy, color=C2, s=22, zorder=2, label="Leaky protocol")
ax.set_yticks(yy, lk.lab)
ax.set_xlabel("Macro-AUC (cross-validated)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
savefig(fig, "fig_leakage")

# ------------------------------------------------------------------ Figure 5: calibration + DCA
cal = pd.read_csv(RES / "e5_calibration.csv")
cur = pd.read_csv(RES / "e5_curves.csv")
dca = pd.read_csv(RES / "e5_dca.csv")
t7 = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
      "Model & Setting & $n$ & Obs. G3 & Mean pred. & Intercept & Slope & G3-AUC \\\\", "\\midrule"]
for _, r in cal.iterrows():
    t7.append(f"{LABELS[r.model]} & {r.set} & {r.n} & {r.prev:.3f} & {r.mean_pred:.3f} & {r.cal_int:+.2f} & {r.cal_slope:.2f} & {r.auc:.3f} \\\\")
t7 += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_calibration", "\n".join(t7))
cx = cal.set_index(["model", "set"])
mac("CalIntMskRaw", cx.loc[("mlr", "MSK-2018 (external)"), "cal_int"], "{:+.2f}")
mac("CalSlopeMskRaw", cx.loc[("mlr", "MSK-2018 (external)"), "cal_slope"], "{:.2f}")
mac("CalIntMskRec", cx.loc[("mlr", "MSK-2018 (intercept-updated)"), "cal_int"], "{:+.2f}")
mac("CalSlopeInt", cx.loc[("mlr", "METABRIC (internal CV)"), "cal_slope"], "{:.2f}")
mac("CalSlopeIntXgb", cx.loc[("xgb", "METABRIC (internal CV)"), "cal_slope"], "{:.2f}")
mac("PrevMsk", cx.loc[("mlr", "MSK-2018 (external)"), "prev"], "{:.2f}")
mac("PredMsk", cx.loc[("mlr", "MSK-2018 (external)"), "mean_pred"], "{:.2f}")
mac("CalIntDukeRaw", cx.loc[("mlr", "Duke MRI (external)"), "cal_int"], "{:+.2f}")
mac("CalSlopeDukeRaw", cx.loc[("mlr", "Duke MRI (external)"), "cal_slope"], "{:.2f}")
mac("CalIntDukeRec", cx.loc[("mlr", "Duke MRI (intercept-updated)"), "cal_int"], "{:+.2f}")
mac("PrevDuke", cx.loc[("mlr", "Duke MRI (external)"), "prev"], "{:.2f}")
mac("PredDuke", cx.loc[("mlr", "Duke MRI (external)"), "mean_pred"], "{:.2f}")
mac("CalSlopeMskXgb", cx.loc[("xgb", "MSK-2018 (external)"), "cal_slope"], "{:.2f}")
for s_, k in [("METABRIC (internal CV)", "int"), ("MSK-2018 (external)", "msk"), ("MSK-2018 (intercept-updated)", "mskrec"),
              ("Duke MRI (external)", "duke"), ("Duke MRI (intercept-updated)", "dukerec")]:
    c = cur[(cur.model == "mlr") & (cur.set == s_)]
    c = c.iloc[np.linspace(0, len(c) - 1, min(len(c), 120)).astype(int)]
    pgf(f"cal_{k}", pd.DataFrame({"p": c.p.to_numpy(), "obs": c.obs.to_numpy()}))
_d = {}
for s_, k in [("METABRIC (internal CV)", "int"), ("MSK-2018 (external)", "msk"), ("Duke MRI (external)", "duke")]:
    d = dca[(dca.model_name == "mlr") & (dca.set == s_)].set_index("threshold")
    _d[f"model_{k}"], _d[f"all_{k}"] = d.model, d.treat_all
pgf("dca", pd.DataFrame(_d).rename_axis("threshold").reset_index())
oe = pd.read_csv(RES / "e5_oe.csv")
t7b = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
       "Model & Setting & Obs. G1 & Pred. G1 & Obs. G2 & Pred. G2 & Obs. G3 & Pred. G3 \\\\", "\\midrule"]
for _, r in oe.iterrows():
    t7b.append(f"{LABELS[r.model]} & {r.set} & " + " & ".join(f"{pct(r[f'obs_g{k}'])} & {pct(r[f'pred_g{k}'])}" for k in (1, 2, 3)) + " \\\\")
t7b += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_oe", "\n".join(t7b))

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
sets = [("METABRIC (internal CV)", C1), ("MSK-2018 (external)", C2), ("MSK-2018 (intercept-updated)", C3),
        ("Duke MRI (external)", C4)]
for s_, col in sets:
    c = cur[(cur.model == "mlr") & (cur.set == s_)]
    axes[0].plot(c.p, c.obs, color=col, lw=2, label=s_)
axes[0].plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (2, 2)))
axes[0].set_xlabel("Predicted P(grade 3)")
axes[0].set_ylabel("Observed proportion (LOWESS)")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=6.5, loc="upper left")
axes[0].set_title("Calibration, multinomial LR", fontsize=7.5)
for s_, col in sets[:2]:
    d = dca[(dca.model_name == "mlr") & (dca.set == s_)]
    axes[1].plot(d.threshold, d.model, color=col, lw=2, label=f"Model — {s_.split(' (')[0]}")
    axes[1].plot(d.threshold, d.treat_all, color=col, lw=1, ls=(0, (3, 2)), label=f"Treat all — {s_.split(' (')[0]}")
axes[1].axhline(0, color=INK2, lw=0.8)
axes[1].set_xlim(0.05, 0.85)
axes[1].set_ylim(-0.05, 0.7)
axes[1].set_xlabel("Threshold probability for grade 3")
axes[1].set_ylabel("Net benefit")
axes[1].legend(fontsize=6.2)
axes[1].set_title("Decision curves, multinomial LR", fontsize=7.5)
fig.tight_layout()
savefig(fig, "fig_calibration_dca")

# net benefit at clinically plausible thresholds
for s_, k in [("METABRIC (internal CV)", "Int"), ("MSK-2018 (external)", "Ext"), ("Duke MRI (external)", "Duke")]:
    d = dca[(dca.model_name == "mlr") & (dca.set == s_)].set_index("threshold")
    adv = d.model - np.maximum(d.treat_all, 0)
    useful = adv[adv > 0.005].index
    mac(f"DcaLo{k}", useful.min() if len(useful) else np.nan, "{:.2f}")
    mac(f"DcaHi{k}", useful.max() if len(useful) else np.nan, "{:.2f}")

# ------------------------------------------------------------------ Table 8 + Figure 6: SHAP & ORs
sh = pd.read_csv(RES / "e6_shap.csv").sort_values("mean_abs_shap", ascending=False)
fn = {"age": "Age", "postmeno": "Post-menopausal", "er": "ER positive", "pr": "PR positive", "her2": "HER2 positive",
      "hist_ilc": "ILC (vs IDC)", "hist_mixed": "Mixed (vs IDC)", "hist_other": "Other type (vs IDC)",
      "t_cat": "T category", "n_pos": "Node positive"}
orr = pd.read_csv(RES / "e6_ordinal_or.csv").set_index("term")
t8 = ["\\begin{tabular}{lrrlrl}", "\\toprule",
      "Feature & mean$|$SHAP$|$ & Direction $\\rho$ (95\\% CI) & Expected & Agrees & Ordinal OR (95\\% CI) \\\\", "\\midrule"]
for _, r in sh.iterrows():
    e = {1: "$\\uparrow$", -1: "$\\downarrow$"}.get(r.expected if not pd.isna(r.expected) else 0, "--")
    ag = "--" if pd.isna(r.agrees) else ("yes" if str(r.agrees) == "True" else "\\textbf{no}")
    o = orr.loc[r.feature]
    unit = " (per 10 y)" if r.feature == "age" else ""
    t8.append(f"{fn[r.feature]}{unit} & {r.mean_abs_shap:.3f} & {r.direction_rho:+.2f} ({r.rho_lo:+.2f}, {r.rho_hi:+.2f}) & "
              f"{e} & {ag} & {o.OR:.2f} ({o.OR_lo:.2f}--{o.OR_hi:.2f}) \\\\")
t8 += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_shap", "\n".join(t8))
pgf("shap", pd.DataFrame({"label": [fn[f] for f in sh.feature][::-1], "y": np.arange(len(sh)),
                          "meanabs": sh.mean_abs_shap.to_numpy()[::-1], "rho": sh.direction_rho.to_numpy()[::-1],
                          "rho_em": (sh.direction_rho - sh.rho_lo).to_numpy()[::-1],
                          "rho_ep": (sh.rho_hi - sh.direction_rho).to_numpy()[::-1]}))
nexp = sh.expected.notna().sum()
nag = (sh.agrees.astype(str) == "True").sum()
mac("ShapAgree", f"{nag} of {nexp}", "{}")
mac("ShapTop", fn[sh.iloc[0].feature], "{}")
mac("ShapSecond", fn[sh.iloc[1].feature], "{}")
mac("OrER", orr.loc["er", "OR"], "{:.2f}")
mac("OrERlo", orr.loc["er", "OR_lo"], "{:.2f}")
mac("OrERhi", orr.loc["er", "OR_hi"], "{:.2f}")
mac("OrILC", orr.loc["hist_ilc", "OR"], "{:.2f}")
mac("OrILClo", orr.loc["hist_ilc", "OR_lo"], "{:.2f}")
mac("OrILChi", orr.loc["hist_ilc", "OR_hi"], "{:.2f}")
mac("OrHer", orr.loc["her2", "OR"], "{:.2f}")
mac("NOrdinal", int(orr.n.iloc[0]), "{:d}")

z = np.load(RES / "e6_shap_values.npz", allow_pickle=True)
sv, Xv, colsv = z["sv3"], z["X"], list(z["cols"])
ordf = list(sh.feature)[::-1]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
rng = np.random.default_rng(0)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
cmap = LinearSegmentedColormap.from_list("fv", [C1, "#d9d8d4", C2])
for i, f in enumerate(ordf):
    j = colsv.index(f)
    x = Xv[:, j]
    xn = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
    ax.scatter(sv[:, j], i + rng.uniform(-0.28, 0.28, len(x)), c=cmap(xn), s=3, alpha=0.6, linewidths=0)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(range(len(ordf)), [fn[f] for f in ordf])
ax.set_xlabel("SHAP value for grade 3 (log-odds contribution), XGBoost")
ax.grid(axis="y", visible=False)
sm_ = plt.cm.ScalarMappable(cmap=cmap)
cb = fig.colorbar(sm_, ax=ax, pad=0.01, fraction=0.03)
cb.set_ticks([0, 1], labels=["low", "high"])
cb.set_label("Feature value", fontsize=7)
savefig(fig, "fig_shap")

# ------------------------------------------------------------------ Table 9 + Figure 7: Saudi transport
tr = pd.read_csv(RES / "e7_transport.csv")
pool = pd.read_csv(RES / "e7_pooled_saudi.csv").set_index("quantity")
agesens = pd.read_csv(RES / "e7_age_sensitivity.csv")
t9 = ["\\begin{tabular}{lllrrrrrr}", "\\toprule",
      "Anchor & Cohort & Quantity & G1 & G2 & G3 (95\\% CI) & ESS & Macro-AUC & G3-AUC \\\\", "\\midrule"]
for (ak, c), g in tr.groupby(["anchor", "cohort"], sort=False):
    r0 = g.iloc[0]
    lab = r0.anchor_label + ("$^\\ddagger$" if ak == "saudi_durrani2025" else "$^\\dagger$")
    t9.append(f"{lab} & {c} & observed & {pct(r0.obs_g1)} & {pct(r0.obs_g2)} & {pct(r0.obs_g3)} & {r0.n_raked} & & \\\\")
    t9.append(f" & & re-weighted & {pct(r0.rw_g1)} & {pct(r0.rw_g2)} & {pct(r0.rw_g3)} ({pct(r0.rw_g3_lo)}--{pct(r0.rw_g3_hi)}) & {r0.ess:.0f} & & \\\\")
    r = g[g.model == "mlr"].iloc[0]
    t9.append(f" & & LR predicted (weighted) & {pct(r.pred_g1)} & {pct(r.pred_g2)} & {pct(r.pred_g3)} & & "
              f"{r.auc_unw:.3f}$\\to${r.auc_w:.3f} & {r.g3auc_unw:.3f}$\\to${r.g3auc_w:.3f} \\\\")
    t9.append(f" & & published & {pct(r0.saudi_g1)} & {pct(r0.saudi_g2)} & {pct(r0.saudi_g3)} & & & \\\\")
    t9.append("\\addlinespace")
t9.append(f"Pooled Saudi anchors (random effects) & & published & {pct(pool.loc['g1','pooled'])} & -- & "
          f"{pct(pool.loc['g3','pooled'])} ({pct(pool.loc['g3','lo'])}--{pct(pool.loc['g3','hi'])}) & & & \\\\")
t9 += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_transport", "\n".join(t9))

def trow(anchor, cohort, model="mlr"):
    return tr[(tr.anchor == anchor) & (tr.cohort == cohort) & (tr.model == model)].iloc[0]

m0 = trow("saudi_kussaibi2021", "METABRIC")
mac("RwGThreeMet", 100 * m0.rw_g3, "{:.1f}")
mac("RwGThreeMetLo", 100 * m0.rw_g3_lo, "{:.1f}")
mac("RwGThreeMetHi", 100 * m0.rw_g3_hi, "{:.1f}")
mac("ObsGThreeMet", 100 * m0.obs_g3, "{:.1f}")
mac("RwGOneMet", 100 * m0.rw_g1, "{:.1f}")
mac("RwGTwoMet", 100 * m0.rw_g2, "{:.1f}")
mac("EssMet", m0.ess, "{:.0f}")
mac("EssRedPct", 100 * (1 - m0.ess / m0.n_raked), "{:.0f}")
mac("PredGThreeMet", 100 * m0.pred_g3, "{:.1f}")
mac("OverPredPP", 100 * (m0.pred_g3 - SA["grade"][3]), "{:.0f}")
mac("NRakedMet", int(m0.n_raked), "{:d}")
mac("AucUnwMet", m0.auc_unw, "{:.3f}")
mac("AucWMet", m0.auc_w, "{:.3f}")
m1 = trow("saudi_kussaibi2021", "MSK-2018")
mac("RwGThreeMsk", 100 * m1.rw_g3, "{:.1f}")
mac("ObsGThreeMsk", 100 * m1.obs_g3, "{:.1f}")
mac("AucWMsk", m1.auc_w, "{:.3f}")
m2 = trow("saudi_kussaibi2021", "Duke MRI")
mac("RwGThreeDuke", 100 * m2.rw_g3, "{:.1f}")
mac("ObsGThreeDuke", 100 * m2.obs_g3, "{:.1f}")
m3 = trow("saudi_durrani2025", "METABRIC")
mac("RwGThreeMetB", 100 * m3.rw_g3, "{:.1f}")
mac("RwGThreeMetBLo", 100 * m3.rw_g3_lo, "{:.1f}")
mac("RwGThreeMetBHi", 100 * m3.rw_g3_hi, "{:.1f}")
mac("SaudiGThreeB", 100 * SB["grade"][3], "{:.1f}")
mac("PoolGThree", 100 * pool.loc["g3", "pooled"], "{:.1f}")
mac("PoolGThreeLo", 100 * pool.loc["g3", "lo"], "{:.1f}")
mac("PoolGThreeHi", 100 * pool.loc["g3", "hi"], "{:.1f}")
mac("PoolK", int(pool.loc["g3", "k_studies"]), "{:d}")
mac("PoolN", int(pool.loc["g3", "n_total"]), "{:d}")
mac("PoolGThreeAll", 100 * pool.loc["g3_all", "pooled"], "{:.1f}")
mac("PoolGThreeAllLo", 100 * pool.loc["g3_all", "lo"], "{:.1f}")
mac("PoolGThreeAllHi", 100 * pool.loc["g3_all", "hi"], "{:.1f}")
a_ = agesens[(agesens.anchor == "saudi_kussaibi2021")]
mac("AgeSensMin", 100 * a_.rw_g3.min(), "{:.1f}")
mac("AgeSensMax", 100 * a_.rw_g3.max(), "{:.1f}")
mac("AgeSensLo", 100 * a_.age_le50.min(), "{:.0f}")
mac("AgeSensHi", 100 * a_.age_le50.max(), "{:.0f}")
mac("RwGThreeAllMin", 100 * tr.rw_g3.min(), "{:.1f}")
_k = tr[(tr.anchor == "saudi_kussaibi2021") & (tr.model == "mlr")].set_index("cohort")
_sh = 100 * (_k.rw_g3 - _k.obs_g3)
mac("ShiftMin", _sh.min(), "{:+.1f}")
mac("ShiftMax", _sh.max(), "{:+.1f}")
mac("ObsGThreeMin", 100 * _k.obs_g3.min(), "{:.1f}")
mac("ObsGThreeMax", 100 * _k.obs_g3.max(), "{:.1f}")
mac("RwGThreeKMin", 100 * _k.rw_g3.min(), "{:.1f}")
mac("RwGThreeKMax", 100 * _k.rw_g3.max(), "{:.1f}")
mac("SaudiGThreeA", 100 * SA["grade"][3], "{:.1f}")
mac("RwGThreeAllMax", 100 * tr.rw_g3.max(), "{:.1f}")

rows_ = []
for ak, sa_ in [("saudi_kussaibi2021", SA), ("saudi_durrani2025", SB)]:
    tag = "$^\\ddagger$" if ak == "saudi_durrani2025" else "$^\\dagger$"
    for c in ["METABRIC", "MSK-2018", "Duke MRI"]:
        r = trow(ak, c)
        rows_.append((f"{c} $\\to$ {sa_['label'].split(' (')[0]}", [r.rw_g1, r.rw_g2, r.rw_g3], r.rw_g3_lo, r.rw_g3_hi))
    rows_.append((f"{sa_['label']} published{tag}", [sa_["grade"][1], sa_["grade"][2], sa_["grade"][3]], np.nan, np.nan))
pgf("transport", pd.DataFrame({"label": [r[0] for r in rows_], "y": np.arange(len(rows_))[::-1],
                               "g1": [100 * r[1][0] for r in rows_], "g2": [100 * r[1][1] for r in rows_],
                               "g3": [100 * r[1][2] for r in rows_]}))
fig, ax = plt.subplots(figsize=(6.2, 3.0))
yy = np.arange(len(rows_))[::-1]
for y_, (l, v, lo, hi) in zip(yy, rows_):
    left = 0
    for k in range(3):
        ax.barh(y_, v[k], left=left, color=SEQ[k], height=0.62, edgecolor="white", linewidth=1.5)
        if v[k] > 0.07:
            ax.text(left + v[k] / 2, y_, f"{100*v[k]:.0f}", ha="center", va="center", color=INK if k == 0 else "white", fontsize=7)
        left += v[k]
ax.set_yticks(yy, [r[0].replace("$\\to$", "→") for r in rows_])
ax.set_xlim(0, 1)
ax.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
ax.grid(axis="y", visible=False)
for k, l in enumerate(["Grade 1", "Grade 2", "Grade 3"]):
    ax.bar(0, 0, color=SEQ[k], label=l)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
ax.axhline(3.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
savefig(fig, "fig_transport")

ta = ["\\begin{tabular}{rrrrr}", "\\toprule", "Assumed share $\\le$50 y & ESS & G1 & G2 & G3 \\\\", "\\midrule"]
for _, r in a_.iterrows():
    ta.append(f"{100*r.age_le50:.0f}\\% & {r.ess:.0f} & {pct(r.rw_g1)} & {pct(r.rw_g2)} & {pct(r.rw_g3)} \\\\")
ta += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_agesens", "\n".join(ta))

# ------------------------------------------------------------------ multiple imputation (R4)
mi = pd.read_csv(RES / "e2mi.csv")
tm = ["\\begin{tabular}{llrrrrr}", "\\toprule", "Model & Missing-data method & $n$ & Macro-AUC (95\\% CI) & G3-AUC & Macro-F1 & Brier \\\\", "\\midrule"]
for _, r in mi.iterrows():
    tm.append(f"{LABELS[r.model]} & {r.method} & {r.n} & {ci(r.macro_auc, r.macro_auc_lo, r.macro_auc_hi)} & {r.g3_auc:.3f} & {r.macro_f1:.3f} & {r.brier:.3f} \\\\")
tm += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_mi", "\n".join(tm))
mim = mi[mi.method.str.startswith("Multiple")].set_index("model")
mac("MIDiffMax", (mim.macro_auc - mim.si_macro_auc).abs().max(), "{:.3f}")
mac("MIMlrAuc", mim.loc["mlr", "macro_auc"], "{:.3f}")
mac("MIXgbAuc", mim.loc["xgb", "macro_auc"], "{:.3f}")
ccm = mi[mi.method.str.startswith("Complete")].set_index("model")
mac("NComplete", int(ccm.loc["mlr", "n"]), "{:d}")
mac("CCMlrAuc", ccm.loc["mlr", "macro_auc"], "{:.3f}")
mac("MIm", cfg_m := yaml.safe_load(open(ROOT / "configs" / "experiment.yaml"))["mi"]["m"], "{:d}")

# ------------------------------------------------------------------ Duke radiomics (R10 / E9)
e9 = pd.read_csv(RES / "e9_duke_radiomics.csv").set_index("design")
nm = {"clinical": "Clinical (full set)", "radiomics": "DCE-MRI radiomics (529 features)", "fusion": "Clinical + radiomics"}
tr9 = ["\\begin{tabular}{lrrrrr}", "\\toprule",
       "Model (L2 logistic) & Features & Macro-AUC (95\\% CI) & G3-AUC (95\\% CI) & $\\Delta$ macro-AUC vs clinical (95\\% CI) & Brier \\\\", "\\midrule"]
for d_, r in e9.iterrows():
    dd = "ref." if d_ == "clinical" else f"{r.d_macro_auc:+.3f} ({r.d_macro_auc_lo:+.3f}, {r.d_macro_auc_hi:+.3f})"
    tr9.append(f"{nm[d_]} & {int(r.n_features)} & {ci(r.macro_auc, r.macro_auc_lo, r.macro_auc_hi)} & {ci(r.g3_auc, r.g3_auc_lo, r.g3_auc_hi)} & {dd} & {r.brier:.3f} \\\\")
tr9 += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_duke_radiomics", "\n".join(tr9))
pgf("duke_radiomics", pd.DataFrame({"label": [nm[d] for d in e9.index], "y": np.arange(len(e9))[::-1],
                                    "macro": e9.macro_auc, "macro_em": e9.macro_auc - e9.macro_auc_lo,
                                    "macro_ep": e9.macro_auc_hi - e9.macro_auc, "g3": e9.g3_auc,
                                    "g3_em": e9.g3_auc - e9.g3_auc_lo, "g3_ep": e9.g3_auc_hi - e9.g3_auc}))
for d_, k in [("clinical", "Clin"), ("radiomics", "Rad"), ("fusion", "Fus")]:
    mac(f"Duke{k}Auc", e9.loc[d_, "macro_auc"], "{:.3f}")
    mac(f"Duke{k}AucLo", e9.loc[d_, "macro_auc_lo"], "{:.3f}")
    mac(f"Duke{k}AucHi", e9.loc[d_, "macro_auc_hi"], "{:.3f}")
    mac(f"Duke{k}GThree", e9.loc[d_, "g3_auc"], "{:.3f}")
for d_, k in [("radiomics", "Rad"), ("fusion", "Fus")]:
    mac(f"Duke{k}Delta", e9.loc[d_, "d_macro_auc"], "{:+.3f}")
    mac(f"Duke{k}DeltaLo", e9.loc[d_, "d_macro_auc_lo"], "{:+.3f}")
    mac(f"Duke{k}DeltaHi", e9.loc[d_, "d_macro_auc_hi"], "{:+.3f}")
mac("NDukeRad", int(e9.loc["clinical", "n"]), "{:d}")

# ------------------------------------------------------------------ sample size (R7 / E10)
ss = json.load(open(RES / "sample_size.json"))
sp = pd.read_csv(RES / "e10_sample_size_pairs.csv")
tss = ["\\begin{tabular}{lrrrrr}", "\\toprule",
       "Sub-model & Share of patients & Anticipated $R^2_{CS}$ & Max $R^2_{CS}$ & $n$ (criterion i) & $n$ (criterion ii) \\\\", "\\midrule"]
for _, r in sp.iterrows():
    tss.append(f"{r.pair} & {r.prop_pair:.3f} & {r.r2cs:.3f} & {r.max_r2cs:.3f} & {np.ceil(r.n_crit1):.0f} & {np.ceil(r.n_crit2):.0f} \\\\")
tss += ["\\midrule", f"Criterion (iii): each grade prevalence $\\pm$0.05 & & & & \\multicolumn{{2}}{{r}}{{{ss['n_crit3']:.0f}}} \\\\",
        f"Required (maximum) / available & & & & \\multicolumn{{2}}{{r}}{{{ss['n_required']:.0f} / {ss['n_available']}}} \\\\",
        "\\bottomrule", "\\end{tabular}"]
save_tab("tab_samplesize", "\n".join(tss))
mac("SSRequired", ss["n_required"], "{:.0f}")
mac("SSAvailable", ss["n_available"], "{:d}")
mac("SSParams", ss["n_params"], "{:d}")
mac("SSMet", "met" if ss["met"] else "not met", "{}")

# ------------------------------------------------------------------ Ljubljana + hyper-params + run info
lj = pd.read_csv(RES / "e8_ljubljana.csv")
t10 = ["\\begin{tabular}{lrrrr}", "\\toprule", "Model & $n$ & Macro-AUC (SD over repeats) & Macro-F1 & G3-AUC \\\\", "\\midrule"]
for _, r in lj.iterrows():
    t10.append(f"{LABELS[r.model]} & {r.n} & {r.macro_auc:.3f} ({r.macro_auc_sd:.3f}) & {r.macro_f1:.3f} & {r.g3_auc:.3f} \\\\")
t10 += ["\\bottomrule", "\\end{tabular}"]
save_tab("tab_ljubljana", "\n".join(t10))
mac("LjBest", lj[lj.model != "majority"].macro_auc.max(), "{:.3f}")
mac("NLj", int(lj.n.iloc[0]), "{:d}")

cfg = yaml.safe_load(open(ROOT / "configs" / "experiment.yaml"))
mac("Repeats", cfg["cv"]["repeats"], "{:d}")
mac("Folds", cfg["cv"]["outer_folds"], "{:d}")
mac("InnerFolds", cfg["cv"]["inner_folds"], "{:d}")
mac("Boot", cfg["bootstrap"], "{:d}")
mac("Seed", cfg["seed"], "{:d}")
mac("TransportBoot", cfg["transport_boot"], "{:d}")
mac("RadRepeats", cfg["radiomics"]["repeats"], "{:d}")

(OUT / "numbers.tex").write_text("% auto-generated by experiments/report.py -- do not edit\n" +
                                 "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in MACROS.items()) + "\n")
print(f"wrote {len(list(OUT.glob('*.tex')))} tex files, {len(list(OUT.glob('*.pdf')))} figures, {len(MACROS)} macros")
