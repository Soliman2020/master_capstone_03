# Project 3 — Applied Machine Learning
### Reducing False Alarms in IoT-Based Fire Detection
![](https://github.com/user-attachments/assets/901e6217-c198-42f3-a6f9-992d446e6ebc)

This project applies supervised machine learning to a public, real-world
multi-sensor IoT dataset to predict fire alarms — with the operational goal of
**reducing false alarms while keeping high recall on real fires**. It is the
first project in the capstone series that produces a *defensible model
evaluation methodology* (rules-first, leakage-aware, regime metrics) carried
into the later agentic and synthesis projects.

---

## TL;DR

- **Dataset:** Smoke Detection Dataset (Kaggle `deepcontractor/smoke-detection-dataset`) — 62,630 rows × 13 sensor features + binary target `Fire Alarm`.
- **Models:** threshold rule, Random Forest, Gradient Boosting, Logistic Regression, Isolation Forest.
- **Headline result:** a random train/test split reports **F1 ≈ 0.9998** for the tree models — but a built-in **leakage audit** shows that number is an artifact. Under **Leave-One-Session-Out (LOSO) cross-validation**, Random Forest scores **0.000 sensitivity** on held-out fire sessions (it misses 100% of alarms it hasn't seen), while the simple **threshold rule transfers best** (sensitivity 0.57, specificity 0.999 on the held-out fire session).
- **Takeaway:** the models learn *session-specific sensor signatures*, not transferable fire signatures. The binding constraint is data diversity, not the algorithm.

---

## Repository contents

```
project_03_ML/
├── README.md                            # this file
├── requirements.txt                      # pinned environment (pip freeze)
├── .gitignore
├── data/
│   ├── raw/
│      └── smoke_detection_iot.csv       # the dataset (~5.6 MB, shipped with the repo)
├── notebooks/
│   └── modeling.ipynb                     # Tasks 1–6 + "A second look" leakage audit
└── reports/
    ├── Machine_Learning_Analysis_Report.md   # Task 7 report source (APA citations)
    ├── Machine_Learning_Analysis_Report.pdf  # Task 7 deliverable (PDF)
    └── md2pdf.py                              # markdown → PDF converter for the report
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle — [`deepcontractor/smoke-detection-dataset`](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset) |
| Rows | 62,630 |
| Columns | 13 sensor features + `UTC` + `CNT` + target `Fire Alarm` |
| Missing values | None |
| Feature types | All numeric (float64 / int64) |
| Target balance | 71.5% alarm / 28.5% no-alarm |
| File | `data/raw/smoke_detection_iot.csv` (~5.6 MB, shipped with the repo) |

**Sensor features:** Temperature[C], Humidity[%], TVOC[ppb], eCO2[ppm],
Raw H2, Raw Ethanol, Pressure[hPa], PM1.0, PM2.5, NC0.5, NC1.0, NC2.5.

A decisive property (discovered in the notebook's audit): the dataset is
**5 continuous recording sessions** from one device, not 62,630 independent
samples — 62,625 of 62,629 row-to-row gaps are exactly 1 second.

---

## Methodology

**Rules-first → ML upgrade**, with a leakage audit:

1. **Baseline threshold rule** — `Fire Alarm = 1 if Temperature[C] > 15 AND TVOC[ppb] > 200` (thresholds tuned by grid search on F1). Interpretable, cannot memorize.
2. **Random Forest** — 200 trees, `max_depth=15`, `min_samples_leaf=5`.
3. **Gradient Boosting** — scikit-learn `GradientBoostingClassifier`.
4. **Logistic Regression** — linear baseline (scale-sensitive, behind a `StandardScaler` pipeline).
5. **Isolation Forest** — *unsupervised* anomaly detector (trained on the full scaled training set, never given the label).

**Evaluation protocol:**
- **Initial (naive) procedure:** stratified 80/20 random split → accuracy/precision/recall/F1 table. (Produces the misleading 0.9998.)
- **"A second look" (leak-proof):** Leave-One-Session-Out CV over the 4 independent recording sessions (duplicate fire sessions 1+2 collapsed into one unit), reporting per-session **sensitivity** and **specificity** — the regime metrics that map onto the false-alarm-reduction goal.

---

## Results

### Naive split (leaked — shown for context)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.9997 | 0.9997 | 0.9999 | 0.9998 |
| Gradient Boosting | 0.9998 | 0.9998 | 0.9999 | 0.9998 |
| Logistic Regression | 0.9268 | 0.9268 | 0.9745 | 0.9501 |
| Baseline rule | 0.6390 | 0.8962 | 0.5596 | 0.6890 |
| Isolation Forest | 0.2684 | 0.3450 | 0.0265 | 0.0492 |

### LOSO (leak-proof — the honest numbers)

| Model | Held-out fire session: sensitivity | Held-out quiet session: specificity |
|---|---|---|
| **Baseline rule** | **0.566** | 0.667 / 0.857 |
| Random Forest | 0.000 | 0.903 |
| Gradient Boosting | 0.264 | 0.000 (fires on everything) |
| Logistic Regression | 0.000 | 0.971 |
| Isolation Forest | 0.000 | bimodal (0 or 1.0) |

**Honest conclusion:** with leakage removed, the simple threshold rule transfers
best across sessions. The ML models overfit to the recording they were trained
on — exactly what produced the naive section's 0.9998. Full per-session table
and the audit code are in the notebook.

---

## How to run

### Prerequisites

- Python 3.11+ (developed on the project-local `p3_venv`)
- Packages: see `requirements.txt` (pinned). Core deps: `scikit-learn`, `pandas`,
  `numpy`, `matplotlib`, `scipy`, `jupyter`.

### 1. Set up the environment

```bash
# from the project root:
# Windows (PowerShell)
. .\p3_venv\Scripts\Activate.ps1
# Linux / macOS
source p3_venv/bin/activate

# install pinned deps if starting fresh
python -m pip install -r requirements.txt
```

### 2. Run the notebook

The dataset is already in `data/raw/` — no download step.

```bash
# interactive:
jupyter lab notebooks/modeling.ipynb

```

Expected runtime: ~2–3 minutes. All cells should execute without error.

**Sanity check after running:** the "A second look" LOSO table should show
**Random Forest sensitivity = 0.000** on the held-out fire session and
**Baseline rule sensitivity ≈ 0.566** on the same session. If those numbers
differ, the session-id assignment in the audit cell has drifted and should be
re-checked.

---

## Project structure & status

| Deliverable | Status |
|---|---|
| `notebooks/modeling.ipynb` | ✅ complete (Tasks 1–6 + leakage audit) |
| `data/raw/smoke_detection_iot.csv` | ✅ shipped |
| `requirements.txt` | ✅ generated from `p3_venv` via `pip freeze` |
| `reports/Machine_Learning_Analysis_Report.pdf` | ✅ generated (Task 7) |


---

## Key finding for the capstone

The leakage audit is itself the most defensible part of this project: it
finds and corrects an evaluation trap that a single-number metric would have
hidden. The transferable artifact for the later agentic (P6) and synthesis (P7)
projects is **methodology + the threshold rule**, not a frozen ML model:

- **Threshold rule** — transferable; expose as the cold-start fire detector.
- **Per-site ML calibration** — train on a site's own history once it exists;
  the in-distribution ~0.9998 is real *for a given site*.
- **Methodology** — rules-first, audit-the-split-for-the-independence-unit,
  report regime metrics, honest failure reporting. P7's fusion risk-scorer
  inherits this discipline on its own synthetic incident data.

Full handoff details are in the root capstone `CLAUDE.md` under
**"P3 → P6/P7 integration note"**.

---

## Limitations & ethics

- **Binding constraint is data diversity, not the algorithm.** Four
  independent recording sessions from one device cannot yield a model that
  generalizes across sites.
- **The 0.9998 F1 was a leakage artifact** from a random split on
  autocorrelated rows — corrected by LOSO.
- **Inherited label bias:** the `Fire Alarm` label was assigned by the
  prototype hardware, not independent ground truth; any model inherits that
  definition of "fire signature."
- **Single-device, single-environment sampling bias** — no variation in sensor
  placement, building type, or occupant behavior.
- **Per-site re-tuning required** — the rule's thresholds reflect this
  prototype and would need calibration in a different environment.

Detailed discussion in `reports/Machine_Learning_Analysis_Report.pdf`
(Sections 6.1–6.3).

---

## Citation

If you reference this work, please cite the dataset and the rubric framework:

- deepcontractor. (2022). *Smoke detection dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset

Full academic references (Breiman 2001, Friedman 2001, Liu et al. 2008,
Roberts et al. 2017, Mehrabi et al. 2021, etc.) are in the report's
**References** section.

---

## License & usage

Educational project for the Udacity AI Mastery Capstone. The dataset is
publicly available under its Kaggle terms; this repo ships the CSV for
reproducibility only. No biometric or PII data is used — the system operates
on sensor metadata only.