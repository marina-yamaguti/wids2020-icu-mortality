# Slide outline — ICU mortality prediction (WiDS Datathon 2020)

10-minute talk, 9 slides, 3 speakers × ~3 min 20 s. All numbers come from the executed `project.ipynb`.
Figures to export from the notebook are indicated as **[Fig: section]**.

| Speaker | Slides | Topic | Time |
|---|---|---|---|
| [Member 1] | 1–3 | Task, metrics, state of the art, our angle | ~3:20 |
| [Member 2] | 4–6 | Data exploration, leakage, validation, models, experiments | ~3:20 |
| [Member 3] | 7–9 | Test results, APACHE comparison, explanation, errors, conclusion | ~3:20 |

---

## [Member 1]

### Slide 1 — Title and task (~1:00)
- Title: *Predicting in-hospital death from the first 24 hours in the ICU*. Three names. Course.
- One sentence: "From the first day of an ICU stay, estimate the probability that the patient dies before leaving hospital."
- 91,713 ICU stays, 186 columns, 147 hospitals; target `hospital_death`, **8.6% positives**.

**Speaker notes:** Why it matters: scarce ICU resources; existing scores (APACHE) are fixed formulas; GOSSIS wants an open, global alternative. Our features are cheap routine measurements, while the target is only known at discharge.

### Slide 2 — Metrics (~0:50)
- Accuracy is useless: "always survives" = 91.4%.
- ROC-AUC (ranking, the Kaggle metric) · PR-AUC (focus on the rare class; chance = 0.086) · recall/precision/alert rate at a clinical threshold · Brier score and calibration · net benefit.

**Speaker notes:** Explain ROC-AUC in one sentence ("probability that a random death is ranked above a random survivor") and why PR-AUC is harder with 9% positives.

### Slide 3 — State of the art and our angle (~1:30)
- Leaderboard: 951 teams; winners 0.91497 / 0.91485 / 0.91445; 50th place 0.91013 (top 50 within 0.005).
- Winners' recipe: big ensembles, stacking, pseudo-labelling, test-time augmentation; most dropped APACHE and IDs.
- Missing from all write-ups: calibration, decision threshold, subgroups, comparison with APACHE.
- **Our question: "Would this model be trustworthy in a hospital it has never seen?"** Only 9 of 66 Kaggle test hospitals appear in training.

**Speaker notes:** Be explicit that we do not compete on the leaderboard and that our test number is not comparable (different patients). Our originality = the evaluation design, not a new algorithm.

**Likely questions (Member 1)**
- *Why not just copy the winning ensemble?* — The gain is ~0.01 AUC, smaller than our fold-to-fold noise (±0.005) combined with the cost in explainability; the course asks for an original, explained solution.
- *Why ROC-AUC if the data are imbalanced?* — ROC-AUC is insensitive to prevalence and is the competition metric (comparability); we complement it with PR-AUC and threshold metrics, which do reflect imbalance.
- *What is APACHE IVa?* — A clinical severity score (logistic-regression formula) computed from the worst first-day values, age, diagnosis and chronic conditions; it outputs a death probability.

---

## [Member 2]

### Slide 4 — Data exploration and cleaning (~1:10)
- Missing values everywhere: 74 columns > 50% missing, 99.97% of rows incomplete. First-hour labs ~92% missing.
- **Missingness is informative**: fewest missing values → ~15% mortality; many missing → ~4% **[Fig: 2.3 right panel]**.
- Cleaning: 1,126 inverted min/max swapped, 778 negative pre-ICU stays → missing, APACHE `-1` → missing, constant column dropped, diagnosis codes → 112 categories. Extremes already clipped by the provider → no outlier removal.

**Speaker notes:** Stress that every decision is justified, e.g. we do not delete outliers because they are the sickest patients.

### Slide 5 — Leakage and validation design (~1:10)
- IDs dropped; `patient_id` unique (no repeated patients).
- Hospitals: mortality from 0% to 29%; the Kaggle test set is mostly new hospitals → **grouped split by hospital**: test set = 34 unseen hospitals (18,312 stays); 5-fold `StratifiedGroupKFold` on the remaining 113 hospitals.
- APACHE predictions = output of another model → excluded from features, kept as benchmark.
- Preprocessing inside scikit-learn pipelines (fit on training folds only).

### Slide 6 — Models and experiments (~1:00)
- Grouped CV ROC-AUC / PR-AUC: Dummy 0.500 / 0.086 · Logistic regression 0.890 / 0.531 · Random forest 0.887 / 0.535 · **LightGBM 0.908 / 0.590** **[Fig: 3.6 strip plot]**.
- Four experiments (one line each):
  1. Random vs grouped split: small optimism (0.909 vs 0.908), but `hospital_id` looks best with a random split (PR 0.600) and worst with a grouped one (0.584).
  2. APACHE as features: no gain (0.905 vs 0.908).
  3. Missingness only: ROC-AUC 0.765 without any measured value.
  4. Class weights: same AUC but mean predicted risk 0.229 vs true 0.086 → no re-weighting.

**Likely questions (Member 2)**
- *Why StratifiedGroupKFold and not GroupKFold?* — It keeps whole hospitals together **and** keeps a similar death rate in each fold, which stabilises PR-AUC.
- *Why does LightGBM handle missing values?* — At each split it learns which side the missing values should go to, using the training data.
- *Why add missing indicators for logistic regression?* — Median imputation alone would make "not measured" look like "normal value"; the indicator lets the model use the fact that a test was not ordered.
- *How did you tune hyperparameters?* — Random search, 6/4/8 candidates, 3-fold grouped CV scored by ROC-AUC; the selected setting is re-evaluated with 5-fold grouped CV.
- *Why is the random-split optimism so small?* — Physiology generalises well across these hospitals; the risk shows up as soon as hospital-specific features are used.
- *Isn't the missingness signal a kind of leakage?* — No: it is available at prediction time (the tests ordered in the first 24 h). But it reflects clinicians' behaviour, which can change, so it must be monitored.

---

## [Member 3]

### Slide 7 — Results on 34 unseen hospitals and comparison with APACHE (~1:20)
- LightGBM: **ROC-AUC 0.900 [0.892–0.908], PR-AUC 0.585 [0.562–0.610], Brier 0.052**.
- Threshold for ≥ 80% recall chosen on development data → **80.0% recall on test**, 22.4% of patients alerted, precision 0.31.
- vs APACHE IVa (15,817 patients with a score): ROC-AUC 0.903 vs 0.862, difference CI [+0.035, +0.048]; for the same recall, 22.1% alerts vs 31.4%.
- **[Fig: 4.5 decision curve]** and **[Fig: 4.4 calibration]**.

**Speaker notes:** Explain the decision curve: above "alert everyone" and "alert no one" = useful; LightGBM above APACHE for all thresholds 2–50%. APACHE over-estimates risk in this population.

### Slide 8 — What drives predictions, and who is missed (~1:20)
- **[Fig: 4.6 SHAP beeswarm]**: diagnosis group, age, ventilation, urea, GCS, SpO2, blood pressure, lactate, low heart rate.
- Missed deaths (317): median predicted risk 0.04, less ventilated (34% vs 73%), normal GCS, fewer tests → they looked stable on day 1. Metabolic deaths most missed (64% of 25), sepsis least (11%).
- Subgroups: AUC falls with age (0.94 → 0.85); low recall in low-risk groups (age < 45: 0.66; cardiac-surgery ICU: 0.55); sex similar; per-hospital AUC 0.78–0.95.

### Slide 9 — Conclusion (~0:40)
- Worked: simple, leakage-aware LightGBM; calibrated; beats APACHE; threshold transfers to new hospitals.
- Limits: day-1 data cannot see later deterioration; global threshold under-alerts low-risk groups; hospital variability → local check/recalibration before use; not comparable with the leaderboard.
- More time: clinical feature engineering, leave-one-hospital-out, recalibration for new hospitals, subgroup thresholds, external validation (MIMIC/eICU).

**Likely questions (Member 3)**
- *Why is a threshold of 0.089 so low?* — Deaths are rare (8.6%); to catch 80% of them we must flag patients whose risk is around or slightly above the average risk.
- *What exactly is a SHAP value?* — The contribution of a feature to one prediction (in log-odds), such that base value + sum of contributions = model output; computed from game theory (Shapley values), exactly and quickly for trees with TreeSHAP.
- *Gain importance vs SHAP?* — Gain measures how much a feature reduced the loss during training; SHAP measures its effect on each prediction and gives the direction.
- *Is the model fair?* — Similar across sexes and the main ethnic groups; lower recall for young patients and low-risk ICUs because of the global threshold; small groups (e.g. 38 Asian deaths) are too uncertain to conclude.
- *How was the confidence interval computed?* — Bootstrap: resample the test patients with replacement 500 times, recompute the metric, take the 2.5th–97.5th percentiles; for the APACHE comparison the same resample is used for both models (paired).
- *Why not calibrate with Platt or isotonic?* — The calibration curve was already on the diagonal; we keep it as the first fix if a new hospital drifts.

---

## Concepts every member should be able to explain
1. Class imbalance and why accuracy fails; ROC-AUC vs PR-AUC (and their chance levels).
2. Data leakage; grouped vs random cross-validation; why preprocessing must be fitted inside the CV folds (pipelines).
3. Stratification (`StratifiedGroupKFold`), out-of-fold predictions, the role of a held-out test set.
4. Missing-data mechanisms (informative missingness), median imputation + indicators, native NaN handling in trees.
5. One-hot vs ordinal encoding; why scaling matters for regularised logistic regression (L2, parameter `C`).
6. Random forest (bagging, feature subsampling) vs gradient boosting (sequential trees, learning rate, number of leaves).
7. Hyperparameter tuning with random search and its budget.
8. Decision threshold vs model; precision/recall trade-off; why the threshold is chosen on development data, not on the test set.
9. Calibration (reliability curve, Brier score), why class weights break it; Platt/isotonic as a fix.
10. Decision curve analysis and net benefit.
11. Bootstrap confidence intervals and paired comparisons.
12. SHAP values (additivity, log-odds, beeswarm reading) and their limits (not causal, correlated features).
13. Subgroup evaluation / fairness; why a global threshold can under-alert low-risk groups.
14. What APACHE IVa is and why it is a benchmark rather than a feature.
