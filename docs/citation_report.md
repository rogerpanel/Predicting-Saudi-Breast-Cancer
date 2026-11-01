# Citation verification report (V1), 2026-10-18

**How this was checked.** Shell curl to Crossref/doi.org was blocked (proxy 403). WebFetch failed for every host, including doi.org, pubmed, ebi.ac.uk and smj.org.sa (DNS ENOTFOUND). So I verified each record through web-search results that index PubMed/PMC, publisher pages, institutional repositories, DBLP/NeurIPS, UCI and TCIA. For each entry, the `note` field in `refs_verified.bib` names the record it was checked against. Before submission, run one automated DOI-resolution pass (e.g., `doi2bib`/Crossref) from a machine with open access.

## A. Original entries

| Key | Status | What changed / comment |
|---|---|---|
| curtis2012metabric | verified | added issue 7403 |
| pereira2016metabric | corrected | title wording changed "refine" to "refines" to match the publisher |
| razavi2018msk | corrected | pages 427–438.e6 |
| mbcproject2022 | corrected (supplemented) | kept as the dataset citation. Primary publication added as `jain2023mbcproject` (medRxiv 10.1101/2023.06.07.23291117). It is **not** a Nature Medicine paper. No peer-reviewed version found |
| stephens2012sanger | verified | added issue |
| banerji2012broad | verified | added issue |
| shah2012bccrc | verified | added issue |
| tcga2012brca | verified | added issue 7418 |
| cerami2012cbio | verified | – |
| gao2013cbio | verified | – |
| zwitter1988ljubljana | verified | UCI dataset page confirms the DOI |
| elston1991grade | verified | – |
| rakha2010grade | verified | – |
| collins2024tripodai | verified | – |
| vonelm2007strobe | verified | – |
| kussaibi2021saudi | corrected | DOI added: **10.15537/smj.2021.42.3.20200711** (PMID 33632908) |
| basudan2022registry | verified | – |
| bray2024globocan | verified | – |
| bergsma2013cramer | verified | – |
| benjamini1995fdr | corrected | DOI added 10.1111/j.2517-6161.1995.tb02031.x |
| vanbuuren2011mice | verified | – |
| chawla2002smote | verified | – |
| vandengoorbergh2022imbalance | verified | – |
| vancalster2019calibration | verified | STRATOS group author added |
| vickers2006dca | verified | – |
| lundberg2017shap | verified (pages unverified) | NeurIPS does not print page numbers. Consider removing pages. No DOI exists |
| lundberg2020tree | verified | added issue 1 |
| chen2016xgboost | verified | pages not re-checked against ACM (standard citation) |
| ke2017lightgbm | verified (pages unverified) | full author list added. No DOI exists |
| grinsztajn2022trees | verified | no DOI exists |
| christodoulou2019lr | verified | – |
| riley2020samplesize | verified | – |
| kapoor2023leakage | verified | – |
| bussola2019slipping | corrected | changed from arXiv-only to the peer-reviewed ICPR 2020 Workshops (Springer LNCS, 2021, pp. 167–182). **The Springer DOI and LNCS volume number are UNVERIFIED** |
| **wang2022duke** | **corrected** | The paper exists: Wang S, Wei Y, Li Z, Xu J, Zhou Y. *Breast Cancer Targets Ther* 2022;14:335–342, doi 10.2147/BCTT.S380651. n = 901 (630 training / 271 validation, random split). Radiomics AUC was **0.761 (0.728–0.794) in training and 0.722 (0.664–0.777) in validation** for high vs low histological grade. **Whether it used Duke-Breast-Cancer-MRI is UNVERIFIED.** No retrieved record names the data source, so do not call it a Duke/TCIA study unless the full-text Methods confirm it. It is also unconfirmed whether the binary outcome is grade 3 vs 1–2. The key name is kept but is misleading. Consider renaming it `wang2022bctt` |
| **kim2025grade** | **corrected** | Actual title: "Predicting Nottingham grade in breast cancer digital pathology using a foundation model." Kim JS, Lee JH, Yeon Y, An D, Kim SJ, Noh M-G, Lee S. *Breast Cancer Res* 2025;27:58. Method: UNI features with 14 MIL algorithms on TCGA-BRCA whole-slide images (521 slides with Nottingham scores). Best result: **F1 0.731, multiclass average AUC 0.835**. Correction notice: 10.1186/s13058-025-02047-0 |
| chen2024uni | verified | added issue 3 |
| hollmann2025tabpfn | verified | added issue 8045 |
| alsolami2021kau | verified | – |
| khaled2022cddcesm | verified | article no. 122 not displayed in retrieved records. The DOI is authoritative |
| aldhabyani2020busi | verified | – |
| lee2017cbis | verified | – |
| cui2021cmmd | verified | full author list added |
| saha2018duke | verified | added issue 4 |
| nguyen2023vindr | verified | article no. 277 not displayed in retrieved records. The DOI is authoritative |
| steyerberg2016validation | verified | – |

## B. New references (claim each one supports)

| Key | Status | Claim supported |
|---|---|---|
| rakha2008nhg | verified | In 2,219 operable breast cancers (median follow-up 111 months), Nottingham grade independently predicted breast-cancer-specific and disease-free survival. Supports the case for adding grade to staging (JCO 2008;26:3153–8) |
| boiesen2000grading | verified (first author + others) | Seven Swedish pathology departments graded the same 93 tumours. All seven agreed on only **31%** of cases, and mean **κ = 0.54** ("moderate") |
| longacre2006interobserver | verified (first author + others) | 13 pathologists graded 35 invasive cancers. Agreement on Nottingham grade was **κ = 0.5–0.7**, best for grades I and III (κ ≈ 0.7 each) |
| vandooijeweert2020variation | verified (first author + others) | Nationwide PALGA data on **33,043** invasive breast cancers (2013–2016) showed significant inter- and intra-laboratory variation in grade. Grade drove adjuvant chemotherapy decisions in nearly a third of patients |
| vandooijeweert2022grading | verified | Review: grading keeps strong prognostic value when done well. Reproducibility can be improved through training, feedback, AI and mitosis stains |
| sotiriou2006ggi | verified (first author + others) | The 97-gene genomic grade index separates grade 2 tumours (30–60% of cases) into low- and high-risk groups. HR **3.61 (2.25–5.78)** for relapse in grade 2 |
| wang2022deepgrade | verified (DOI unverified) | DeepGrade, a deep-learning model on H&E whole-slide images, was trained/validated on 1,567 patients and tested externally on 1,262. It split NHG2 into two groups. The DG2-high group had higher recurrence risk (HR **1.91, 1.11–3.29**) |
| wang2026dcemrigrade | verified (volume unverified) | DCE-MRI radiomics in 198 IDC patients (high vs intermediate/low grade). Logistic regression beat random forest, with AUC **0.795** for LR vs 0.700 for RF (which cohort these AUCs come from is not confirmed). Supports the point that LR matches or beats ML for grade prediction |
| pate2023multinomial | verified | Minimum sample-size criteria for multinomial prediction models, using events per outcome category, shrinkage/R² and overall-risk precision. Directly relevant to the three-class grade outcome |
| collins2024evalpart1 | verified | Model evaluation via internal, internal-external and external validation, including heterogeneity and fairness (BMJ series part 1) |
| riley2024evalpart2 | verified | How to run an external validation study: calibration, discrimination, clinical utility |
| riley2024evalpart3 | verified | Sample size for external validation. Many validation studies are too small. Tailored calculations are better than rules of thumb |
| vanhoorde2014multinomial | verified (authors from recall) | Multinomial calibration framework: recalibration intercepts and slopes, plus vector-spline calibration plots for polytomous outcomes |
| riley2023stability | verified | Four levels of prediction instability. Bootstrap instability plots are recommended. Instability shows up as miscalibration in new data |
| rosenblatt2024leakage | verified | Across 4 datasets and 3 phenotypes, leakage from feature selection and repeated subjects inflated performance the most. Small samples were more sensitive to leakage (Nat Commun 2024;15:1829) |
| varoquaux2022failures | verified | Methodological failures in medical ML, including evaluation biases, benchmark overfitting and leakage. Benchmark gains do not translate to clinical gains |
| scr2022report | partially verified | Saudi Cancer Registry 2022: breast cancer is **31.3%** of cancers in Saudi females and **17.3%** of all cancers in Saudi nationals. 3,280 female cases, ASR **49.7/100,000**. These figures come from search-indexed excerpts and a secondary study, so check the report table before quoting. ⚠ The GLOBOCAN-based Saudi breast ASR reported in a secondary analysis was ~25.3/100,000. That figure and the GCO fact-sheet numbers (≈3,865 cases) are **UNVERIFIED** and have no bib entry. Do not mix the registry and GLOBOCAN figures |
| alessy2024gcc | verified | GLOBOCAN-based GCC burden: 42,475 cases in 2020. Female breast is the most common cancer (16%). Burden projected to rise 116% (Saudi Arabia) to 270% (Qatar) by 2040 |
| najjar2010arab | verified | Review of 28 studies from Arab nations: mean age at diagnosis **48 y** (range 43–52), **65.5%** of patients under 50, about a decade younger than in the West. Caveat: a 2024 EMHJ analysis argues this reflects population age structure, not higher age-specific incidence |
| chouchane2013arab | verified | Arab patients are about a decade younger, present at a later stage with larger tumours, and show more node involvement, more HR-negative disease and **higher grade** |
| elkum2007young | verified | 867 Saudi patients at KFSH&RC. **33.2%** were ≤40 years old. Age ≤40 independently predicted relapse. 5-y DFS was 77.4% (≤40) vs 83.5% (>40) |
| altamimi2010saudi | verified | IHC subtyping of 231 Saudi tumours: luminal A 3.9%, luminal B 16%, **HER2 17.3%**, basal 10%, unclassified 42.8% |
| sisk2023imputation | verified (DOI unverified) | Simulation: how to handle the outcome in imputation depends on whether missingness is allowed at deployment. Missing indicators can help, but are harmful when missingness depends on the outcome |
| hoogland2020missing | verified | Imputation and handling of missing predictors at validation and deployment should match the model's intended use, including single-patient imputation |
| steingrimsson2023transport | verified | Tailoring a prediction model and estimating its performance in a target population using only target covariates (outcome-free target). Formal basis for case-mix reweighting/raking |
| debray2015framework | verified | Framework for interpreting external validation through case-mix relatedness, on a spectrum from reproducibility to transportability |
| riley2016bigdata | verified | Big/IPD datasets allow checking calibration and discrimination across settings and subgroups, and support recalibration rather than discarding a model |
| jain2023mbcproject | verified (preprint) | Primary MBCproject description: patient-partnered, online-consent, genomic and clinical data released publicly (medRxiv 2023) |

## C. Items still UNVERIFIED (fix before submission)
1. wang2022duke: whether the data source is Duke/TCIA, and whether the binary coding is grade 3 vs 1–2.
2. bussola2019slipping: Springer DOI and LNCS volume number.
3. wang2022deepgrade: DOI (10.1016/j.annonc.2021.09.007 is from recall).
4. sisk2023imputation: DOI.
5. wang2026dcemrigrade: volume number.
6. lundberg2017shap / ke2017lightgbm: page ranges (consider dropping them).
7. scr2022report: exact figures and report year. GLOBOCAN Saudi fact-sheet numbers (not entered).
8. vanhoorde2014multinomial: author list (from recall; title, journal and DOI are verified).
9. Prior studies using **METABRIC or SEER clinicopathological variables to predict grade**: none found in several searches. All prior grade-prediction work found was image-based (MRI radiomics, WSI deep learning) or gene-expression based. This gap supports the novelty claim, but it is phrased as "we found no", not "none exist".
