# Machine Learning Analysis Report

**Project 3 — Applied Machine Learning (Machine Learning Foundations)**


## 1. Overview

This project applies supervised machine learning to **IoT-based fire detection**,
with the goal of reducing false alarms while maintaining high recall on real
fire events. The dataset used is the **Smoke Detection Dataset**
(Kaggle, `deepcontractor/smoke-detection-dataset`), a real, non-synthetic,
publicly available multi-sensor recording from an IoT fire-detection prototype
([deepcontractor, 2022](See References)). The task is framed as **supervised
binary classification** of the `Fire Alarm` target (0 = no fire, 1 = fire)
from sensor readings. We compare a transparent, hand-tuned threshold
rule against three supervised classifiers (Random Forest, Gradient Boosting,
Logistic Regression) and one unsupervised anomaly detector (Isolation Forest).

The notebook implements the full workflow — problem definition, data loading
and inspection, preprocessing, model training, evaluation, and a written
summary — and additionally contains a self-contained **leakage audit**
("A second look" section) that re-evaluates every model under a
leak-proof cross-session protocol.

## 2. Dataset Description

The Smoke Detection Dataset contains **62,630 rows** and **15 columns** (13
sensor features, a Unix timestamp `UTC`, a sample-count column `CNT`, and the
binary target `Fire Alarm`). There are **no missing values**, and all features
are numeric (float64 or int64), so no imputation or categorical encoding is
required for the raw features.

The sensor features are:

- **Temperature[C]** — ambient temperature in degrees Celsius.
- **Humidity[%]** — relative humidity.
- **TVOC[ppb]** — total volatile organic compounds, in parts per billion.
- **eCO2[ppm]** — equivalent CO₂, in parts per million.
- **Raw H2** and **Raw Ethanol** — raw MOS sensor outputs for hydrogen and
  ethanol.
- **Pressure[hPa]** — atmospheric pressure.
- **PM1.0, PM2.5** — particulate-matter mass concentrations.
- **NC0.5, NC1.0, NC2.5** — number concentrations of particles in the named
  size bins.

The target distribution is **71.5% alarm (`Fire Alarm = 1`) vs. 28.5%
no-alarm**, a moderate class imbalance that motivates the use of
precision/recall/F1 rather than accuracy alone.

A property of this dataset that proved decisive for the evaluation is its **temporal structure**: as inspected in the notebook, 62,625
of the 62,629 row-to-row time gaps are exactly one second, and only four gaps
exceed sixty seconds. The data is therefore a small number of continuous
recording *sessions* from a single device, not 62,630 independent samples.




## 3. Modeling Approach

### 3.1 Preprocessing

Preprocessing follows the principle that every transformation fitted on the
training data must not see the test data (Hastie, Tibshirani, & Friedman,
2009). Concretely:

1. **Feature engineering** — a cyclical time-of-day feature was derived from
   `UTC` using sine and cosine encoding (`hour_sin`, `hour_cos`), so that
   midnight (hour 0) and 11 p.m. (hour 23) are close in feature space rather
   than numerically distant (Kuhn & Johnson, 2013). The raw `UTC` timestamp
   and the `CNT` sample-count column were then dropped: the former is
   represented by the engineered features, and the latter is firmware metadata,
   not a physical signal.
2. **Scaling** — features were standardized with `StandardScaler` fit on the
   training set only. Tree-based models are invariant to monotonic scaling
   (Hastie et al., 2009), so scaling does not affect them; it was retained
   because the Isolation Forest is distance-based and benefits from it, and
   because Logistic Regression is scale-sensitive.
3. **Splitting** — the initial procedure used a stratified 80/20 train/test
   split (`random_state = 42`). This split is the one whose results are
   reported in Section 4.1; Section 4.2 re-evaluates under a leak-proof
   protocol and explains why the two differ.

### 3.2 Models

Five models were implemented, all with scikit-learn (Pedregosa et al., 2011):

1. **Baseline threshold rule** — `Fire Alarm = 1 if Temperature[C] > T AND
   TVOC[ppb] > V`. The thresholds (`T = 15`, `V = 200`) were selected by a
   coarse grid search maximizing F1 on the training set. This is the
   interpretable, "rules-first" reference point.
2. **Random Forest** (Breiman, 2001) — 200 trees, `max_depth = 15`,
   `min_samples_leaf = 5`, `min_samples_split = 10`. Chosen for robustness to
   multicollinearity among the particulate features (PM1.0/PM2.5 and the
   NC\* counts are strongly correlated) and for its feature-importance output.
3. **Gradient Boosting** (Friedman, 2001) — scikit-learn's
   `GradientBoostingClassifier` with default hyperparameters. Boosting
   typically performs well on tabular data (Chen & Guestrin, 2016).
4. **Logistic Regression** — a linear, scale-sensitive baseline.
5. **Isolation Forest** (Liu, Ting, & Zhou, 2008) — an *unsupervised* anomaly
   detector trained on the full scaled training set with
   `contamination = 0.05`, included to test whether alarms can be detected as
   statistical outliers without using the label.

### 3.3 Evaluation metrics

Because the classes are moderately imbalanced and the operational cost of a
false alarm (a needless dispatch) differs from the cost of a missed fire,
**precision, recall, and F1 are the primary metrics**; accuracy is reported
but not treated as decisive (Powers, 2020). For the leak-proof re-evaluation
(Section 4.2), where each held-out recording session is mixed (contains both
alarm and no-alarm rows), we additionally report per-session **sensitivity**
(recall on the session's alarm rows) and **specificity** (the true-negative
rate on the session's no-alarm rows) — the two regime metrics that map directly
onto the false-alarm-reduction goal (Fawcett, 2006).

## 4. Results

### 4.1 Initial procedure (stratified random split)

Under the stratified 80/20 split, the supervised tree models achieve
near-perfect scores on the held-out test set:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.9997 | 0.9997 | 0.9999 | **0.9998** |
| Gradient Boosting | 0.9998 | 0.9998 | 0.9999 | **0.9998** |
| Logistic Regression | 0.9268 | 0.9268 | 0.9745 | 0.9501 |
| Baseline rule | 0.6390 | 0.8962 | 0.5596 | 0.6890 |
| Isolation Forest (unsupervised) | 0.2684 | 0.3450 | 0.0265 | 0.0492 |

Five-fold cross-validation F1 on the full dataset (regular `KFold`) was
0.9395 (±0.0426) for Random Forest, 0.8697 (±0.1180) for Gradient Boosting,
and 0.7319 (±0.2689) for Logistic Regression.

Taken at face value, the tree models appear to solve the problem. Section 4.2
shows this face value is misleading.

### 4.2 Re-evaluation under a leak-proof protocol ("A second look")

A near-perfect F1 on a real-world IoT dataset is a red flag rather than a
triumph. The notebook's audit confirms the data is **five continuous recording
sessions** separated by gaps of more than sixty seconds, two of which (sessions
1 and 2) are the **same fire recorded twice** — identical row count, duration,
and every gas/particulate feature mean; sorted-TVOC correlation = 1.0000. A
random train/test split interleaves adjacent one-second-apart rows across the
boundary; because adjacent rows are near-duplicate readings that almost always
share the same label, a tree model trained on row *t* effectively sees the
answer for row *t+1* in the test set. This is a textbook instance of
autocorrelation-induced data leakage (Roberts et al., 2017).

After collapsing the duplicate sessions into a single cross-validation unit,
there are **four independent sessions**. No leak-free split preserves the
71/29 class balance (the duplicated fire is ~80% of all rows and is entirely
alarms), so the evaluation uses **Leave-One-Session-Out (LOSO)
cross-validation**: each fold trains on three sessions and is scored on the
one held-out session, reporting per-session sensitivity and specificity.

| Model | Session | Type | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| **Baseline rule** | 1 | fire (1+2) | **0.566** | 0.999 | 0.723 |
| Baseline rule | 3 | short fire | 0.426 | 1.000 | 0.597 |
| Baseline rule | 0, 4 | quiet | 0.000 | 0.667 / 0.857 | 0.000 |
| Random Forest | 1 | fire (1+2) | **0.000** | 1.000 | 0.000 |
| Random Forest | 3 | short fire | 0.000 | 1.000 | 0.000 |
| Random Forest | 0 | quiet | 0.000 | 0.903 | 0.000 |
| Random Forest | 4 | quiet | 1.000 | 0.304 | 0.001 |
| Gradient Boosting | 1 | fire (1+2) | 0.264 | 0.998 | 0.417 |
| Gradient Boosting | 3 | short fire | 0.000 | 1.000 | 0.000 |
| Gradient Boosting | 4 | quiet | 1.000 | 0.000 | 0.001 |
| Logistic Regression | 1 | fire (1+2) | 0.000 | 1.000 | 0.000 |
| Logistic Regression | 3 | short fire | 0.121 | 0.970 | 0.216 |
| Isolation Forest | 1 | fire (1+2) | 0.000 | 1.000 | 0.000 |
| Isolation Forest | 3 | short fire | 1.000 | 0.000 | 0.986 |

(Selected rows shown; the full table is the `loso_results` dataframe in the
notebook.)

The picture inverts completely. **Random Forest scores 0.000 sensitivity on
every held-out fire session** — it misses 100% of alarms in sessions it has not
seen — while staying correctly quiet on one quiet session. **Gradient Boosting**
is the only supervised ML model with non-trivial cross-session transfer
(sensitivity 0.264 on the held-out fire unit), but it fires on nearly every row
of the other held-out quiet session (specificity 0.000). The **Isolation
Forest** is bimodal and operationally useless: it either misses all alarms or
fires on everything. Crucially, the **simple threshold rule transfers best** —
on the held-out fire session it reaches sensitivity 0.566 with specificity
0.999, catching more than half the real alarms while staying quiet.

## 5. Interpretation for a Non-Technical Audience

Imagine a smoke detector that, when tested on data it has already seen, appears
to be right 99.98% of the time. That sounds perfect — until you ask how it does
on a recording it has *never* seen. When we tested that way, the most
sophisticated model we built (Random Forest) **missed every single real fire**
in the new recording. It had not learned what a fire looks like; it had
memorized the specific recording it was trained on.

Why did this happen? The dataset is not 62,000 separate, independent
measurements. It is one sensor device recording continuously, one reading per
second, for a handful of long sessions. Two of those sessions are actually the
same fire recorded twice. When the data is split randomly, a reading from
second 1000 lands in the "training" pile and the almost-identical reading from
second 1001 lands in the "test" pile — so the model is effectively being
handed the answer during evaluation. Its 99.98% score measures "can it
recognize this specific recording," not "can it recognize a fire."

The lesson is practical: **a high score is not the same as a working model.**
The honest test — train on some recordings, evaluate on a completely different
one — revealed that the fancy models fail, while a very simple rule
("call it a fire if it is warmer than 15°C *and* the air has more than 200
parts per billion of volatile compounds") actually holds up reasonably well on
unseen data, catching more than half of real fires while rarely crying wolf.

For a real deployment, the takeaway is: don't trust a number until you have
checked whether the test data is genuinely separate from the training data.
And sometimes the simple, explainable rule is the safer thing to ship.

## 6. Limitations and Potential Bias

### 6.1 Limitations and tradeoffs

- **The binding constraint is data diversity, not the algorithm.** The dataset
  is effectively four independent recording sessions from a single device. The
  supervised ML models overfit to session-specific sensor signatures and do
  not generalize across sessions (Section 4.2). No amount of hyperparameter
  tuning on this data will produce a model that transfers to a new building; a
  larger, multi-session, multi-device dataset is required for that.
- **The initial 0.9998 F1 was a leakage artifact.** A random split on
  autocorrelated data inflates model scores (Roberts et al., 2017); the
  honest cross-session numbers are roughly four orders of magnitude lower for
  the tree models. This is both a limitation of the dataset's structure and a
  demonstration of why a single-number evaluation is insufficient.
- **The threshold rule was tuned on the full dataset.** Its thresholds
  (`T = 15`, `V = 200`) reflect this particular prototype and would need
  per-site re-calibration in a different environment.

### 6.2 Potential sources of bias

- **Inherited label bias.** The `Fire Alarm = 1` label was assigned by the
  prototype hardware itself, not by independent ground truth. Any model
  trained on this data inherits the device's definition of a "fire
  signature," which may not match a human or regulatory definition. This is a
  form of label bias that cannot be removed from the data alone.
- **Single-device, single-environment sampling bias.** All recordings come
  from one prototype in one location; the model has seen no variation in
  sensor placement, ambient conditions, building type, or occupant behavior.
  A model deployed elsewhere would face distribution shift it has never
  encountered (Mehrabi, Morstatter, Saxena, Lerman, & Galstyan, 2021).
- **Class-imbalance artifact.** The 71.5% alarm rate is itself an artifact of
  one seven-hour fire (recorded twice) dominating the dataset, not a
  reflection of the true base rate of fires in deployment.

### 6.3 Steps taken or proposed to reduce these risks

- **Leak-proof evaluation.** We replaced the random split with
  Leave-One-Session-Out cross-validation that respects the recording session
  as the independence unit, so the reported numbers reflect genuine
  cross-session generalization rather than in-session memorization.
- **Honest reporting over vanity metrics.** The near-perfect initial score was
  preserved in the notebook *and* corrected, rather than silently replaced, so
  the failure mode is visible and auditable.
- **Proposed for deployment (not yet implemented):** use the interpretable
  threshold rule as a cold-start detector for new sites, train a per-site ML
  model only once that site has accumulated its own history, and route
  low-confidence or distribution-shifted cases to human review. Collecting a
  multi-device, multi-environment dataset is the single highest-leverage
  mitigation for the sampling and label biases above.


## References

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
*Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

deepcontractor. (2022). *Smoke detection dataset* [Data set]. Kaggle.
https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset

Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition
Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting
machine. *The Annals of Statistics*, 29(5), 1189–1232.
https://doi.org/10.1214/aos/1013203451

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of
statistical learning: Data mining, inference, and prediction* (2nd ed.).
Springer. https://doi.org/10.1007/978-0-387-84858-7 https://github.com/tpn/pdfs/blob/master/The%20Elements%20of%20Statistical%20Learning%20-%20Data%20Mining,%20Inference%20and%20Prediction%20-%202nd%20Edition%20(ESLII_print4).pdf

Kuhn, M., & Johnson, K. (2013). *Applied predictive modeling*. Springer.
https://doi.org/10.1007/978-1-4614-6849-3

Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *2008 Eighth
IEEE International Conference on Data Mining*, 413–422.
https://doi.org/10.1109/ICDM.2008.17

Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A
survey on bias and fairness in machine learning. *ACM Computing Surveys*,
54(6), 1–35. https://doi.org/10.1145/3457607

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel,
O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python.
*Journal of Machine Learning Research*, 12, 2825–2830. https://dl.acm.org/doi/10.5555/1953048.2078195

Powers, D. M. W. (2020). Evaluation: From precision, recall and F-measure to
ROC, informedness, markedness and correlation. *arXiv preprint*.
https://arxiv.org/abs/2010.16061

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita,
G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton,
D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation
strategies for data with temporal, spatial, hierarchical, or phylogenetic
structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881

