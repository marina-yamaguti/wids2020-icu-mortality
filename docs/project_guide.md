# Project guide: what changed, and why each choice was made

This guide is for writing the **Observations** of `newproject.ipynb` together. Every number below comes from the run of 22 October (the notebook is saved with these outputs). The "…" in the notebook are left for you: this guide gives the facts and the questions to answer, not the sentences.

---

## Part 1. What was changed on 22 October

| # | Change | Why |
|---|---|---|
| 1 | **Section 3.8 (hospital overfitting):** the table now shows `mean ± std`; the pre-written conclusion was replaced by neutral questions. **Section 4.3:** a new check compares the death rate with the mean predicted risk in training and test hospitals | The old text said `hospital_id` "helps with a random split, not with a grouped one". The run shows differences of about 0.002, smaller than the fold spread (±0.006). The hospital effect is better looked for in the probabilities |
| 2 | **Wider grids:** `C` now starts at 0.0001, `num_leaves` goes up to 127 | Before, the best values (C = 0.001, 63 leaves) were on the edge of the grid, so we couldn't know whether a further value was better. Now both are inside the grid. The old text "bigger trees don't improve validation" was replaced by questions |
| 3 | **Section 3.13 (simple models, smart preprocessing):** new clinical features (`gcs_total`, `shock_index`, five first-day ranges, `n_comorbidities`) tested on both models, and LightGBM with only 30 features (selection inside the pipeline) | It's the teacher's main request, and the section only had two small ablations |
| 4 | **Split kept as `GroupShuffleSplit`** (your choice). Section 2.9 now explains why the test set has 23% of the stays and a higher death rate, with an observation to write | A new hospital never has exactly the training death rate. We keep the difference on purpose and study it in Section 4.3 |
| 5 | **152 → 111 cells:** headings merged with their text, `---` separators removed, small code cells merged (grid search + plot + evaluation in one cell) | Teammate's request. Nothing was removed, and new content was added |
| 6 | **Subgroups added** (Section 4.6): age, sex, ICU type, elective surgery | The project proposal promised it |
| 7 | **Text fixes:** "ROC-AU", "0.5 = mean" → "0.5 = random", a dangling colon, em dashes in headings, references to SHAP and decision curves (no longer used) | Small errors |
| 8 | **Repo:** `project.ipynb`, `wids_utils.py` and `slides_outline.md` moved to `archive/`; README rewritten | Only one notebook can be submitted |
| + | **Bug fix:** the one-hot column names are cleaned (`Operating Room / Recovery` → `Operating_Room_Recovery`) | LightGBM renames columns with spaces, which broke the 30-feature pipeline. Nothing else changes |

Not changed: the slide deck on the Desktop (old numbers) and `docs/presentation_notes.md` (its "…" can now be filled from Part 3). Nothing was committed or pushed.

---

## Part 2. The project, choice by choice

### Our question (Sections 1.1 to 1.4)
Predict in-hospital death from the first 24 h of an ICU stay. The teacher's feedback: don't chase the leaderboard, find the **sources of overfitting**, and get close to the SOTA with **simple models and smart preprocessing**. So the notebook is organised around those two things, and APACHE IVa is a benchmark, not a feature.

- **Why ROC-AUC?** It is the Kaggle metric (comparison with the SOTA) and doesn't depend on the 8.6% death rate. PR-AUC is added because it focuses on the rare class (chance level = death rate, not 0.5).
- **Why not accuracy?** "Always survives" gives 91.37% and detects nobody.

### Preprocessing (Section 2)

| Choice | Why | Alternative we didn't take |
|---|---|---|
| Keep missing values as `NaN` (trees) or impute with the median **+ a "was missing" column** (logistic regression) | 99.97% of the rows have a missing value, and patients with *fewer* missing values die *more* (doctors test more when worried): missingness is information | Drop rows (impossible) or impute without indicators (hides the information) |
| Don't remove outliers | The provider already clipped extreme values; the remaining extremes are the sickest patients | Remove values outside 1.5 IQR (would delete the patients we most want to detect) |
| Fix impossible values: 1,126 inverted min/max swapped, 778 negative pre-ICU stays and 2,371 APACHE `-1` → missing | They are recording errors or codes, not measurements | — |
| Diagnosis codes → categories (integer part) | 502.01 is a label, not a quantity | Keep as numbers (meaningless order) |
| One-hot encoding with `pd.get_dummies`, **before** the split | It doesn't use the target, so it can't leak | Target encoding (uses the target, see Section 3.11) |
| Drop identifiers; `hospital_id` used only to **group** | A new hospital is the real use case | Keep `hospital_id` (see Section 3.8) |
| APACHE IVa probabilities removed from the features | It's another model's output; we compare against it instead | Keep it (tested in 3.12: no gain) |
| Test set = 20% of the **hospitals** (`GroupShuffleSplit`) | No test hospital is ever seen in training | `StratifiedGroupKFold` would balance the death rate; we kept the difference on purpose (change 4) |

### Training (Section 3)

- **Validation:** `GroupKFold` (5 folds by hospital) for every choice, and `cross_validate(..., return_train_score=True)` to see train vs validation. **Train − validation = how much the model memorises.**
- **Models, from simple to flexible:** Dummy (0.5 by construction), logistic regression (linear, needs imputation and scaling), random forest (averaged deep trees), LightGBM (trees added one by one, each correcting the previous ones; used by all top Kaggle teams).
- **Tuning with `GridSearchCV`:** one parameter per model, the one that controls complexity: `C` (regularisation), `min_samples_leaf` (leaf size), `num_leaves` (tree size).
- **Five sources of overfitting** (Sections 3.7 to 3.11), then APACHE as a feature (3.12) and preprocessing tests (3.13).

### Evaluation (Section 4)
The final models are trained once on all the training hospitals and tested once on the 30 test hospitals: ROC/PR, a **threshold chosen on the training set** (out-of-fold predictions) to catch 80% of deaths, calibration, feature importance, APACHE comparison, missed deaths, per hospital, subgroups.

- **Why choose the threshold on the training set?** Choosing it on the test set would use the test labels: the result would be optimistic.
- **Why 80% recall?** A screening tool: missing a death is worse than an extra review. It's our assumption, one parameter (`TARGET_RECALL`).

---

## Part 3. The facts for each observation

Use these to write the "…" in your own words. The questions in the notebook tell you what to discuss.

**2.2 Target:** 7,915 deaths / 91,713 = 8.63%; "always survives" = 91.37% accuracy.

**2.3 Missing values:** 175/186 columns have missing values, 74 more than 50%, 34 more than 80%, 99.97% of rows incomplete. Mortality is highest for the patients with the fewest missing values (≈15%) and lowest around 4%.

**2.5 Easy task?** Best single measured feature: `d1_lactate_max`, ROC-AUC **0.736**. 63 pairs of features with |r| > 0.95.

**2.6 Leakage:** 147 hospitals (median 323 stays); mortality from 0% to 29.1% in hospitals with ≥ 100 stays. Kaggle test file: 66 hospitals, only 9 also in training. APACHE IVa alone: ROC-AUC 0.863 (88.4% of stays have it).

**2.9 Split:** train 70,261 stays, 117 hospitals, mortality **8.13%**; test 21,452 stays (**23%** of the stays), 30 hospitals, mortality **10.26%**.
> To discuss: the test hospitals are sicker than the training ones. Is that a problem, or a realistic situation?

**3.3 to 3.6 Models (5-fold grouped CV)**

| Model | Best setting | Train | Validation ± std | Gap |
|---|---|---|---|---|
| Dummy | — | 0.500 | 0.500 | 0 |
| Logistic regression | C = 0.001 (0.0001 is worse: 0.883) | 0.899 | **0.891 ± 0.007** | 0.008 |
| Random forest | min_samples_leaf = 10 | 0.967 | **0.887 ± 0.009** | 0.080 |
| LightGBM | num_leaves = 63 (31 and 127 are almost equal, 4 is lower: ≈0.900) | 0.999 | **0.906 ± 0.006** | 0.092 |

> To discuss: in the `num_leaves` plot, the training score goes from 0.91 to 1.00 while validation stays around 0.906 from 31 leaves on. What does that say about tree size? Is a difference of 0.0005 between 31, 63 and 127 meaningful compared with ±0.006?

**3.8 Training hospitals**

| | with `hospital_id` | without |
|---|---|---|
| new hospitals (GroupKFold) | 0.9049 ± 0.0069 | 0.9064 ± 0.0062 |
| same hospitals (KFold) | 0.9085 ± 0.0059 | 0.9080 ± 0.0054 |

> To discuss: the random split is 0.0016 more optimistic; `hospital_id` changes the score by less than 0.002. Both are smaller than the ±. The teacher expected overfitting of the training hospitals: for ranking the patients, we see very little. This is a valid, honest result. Section 4.3 looks at the probabilities instead.

**3.9 Number of trees:** validation best at 500 trees (0.9048), then 0.9036 (1,000) and 0.9015 (2,000); training reaches 1.000 from 1,000 trees.

**3.10 Hospital fingerprints:** among the 20 largest hospitals, the hospital is recognised **80.6%** of the time from the features, **41.8%** from the missing-value pattern alone, vs **10.7%** by always guessing the largest. Missing-value pattern alone predicts death: ROC-AUC **0.765 ± 0.021**.
> This is the strongest "hidden overfitting" result: removing `hospital_id` doesn't remove the hospital.

**3.11 Leaky preprocessing:** 8,784 diagnosis × ICU groups. Naive target encoding: **0.946** (above the Kaggle winner, which is impossible: it's fake). Inside the pipeline: **0.905**, i.e. no gain over 0.906.
> Why did the naive version cheat? Each validation patient's own outcome is inside its group's death rate.

**3.12 APACHE as features:** 0.9038 vs 0.9064 without.

**3.13 Preprocessing**

| Test | Validation | Compared with |
|---|---|---|
| LR without missing indicators | 0.8873 | 0.8911 with (+0.004) |
| LightGBM without `n_missing_values` | 0.9057 | 0.9064 with (+0.0007, inside the ±) |
| LR + clinical features | 0.8913 | 0.8911 (no change) |
| LightGBM + clinical features | 0.9063 | 0.9064 (no change) |
| LightGBM, 30 features | 0.8742 | 0.9064 (−0.032) |

> To discuss: the clinical features didn't help. Possible reason: most of them are sums or differences of existing columns (`gcs_total`, the ranges, `n_comorbidities`), which a linear model can already build and trees can approximate; only `shock_index` (a ratio) is really new. The missing indicators are the preprocessing that helps the most. Keeping only 30 features loses 0.03: the information is spread over many features. Our best single model (0.906) is 0.004 below the best single LightGBM reported on Kaggle (0.9097, random split, 700+ features), which is less than one fold std.

**4.1 Test hospitals**

| Model | Train | CV | Test ROC-AUC | Test PR-AUC | Brier |
|---|---|---|---|---|---|
| Logistic regression | 0.899 | 0.891 | **0.882** | 0.528 | 0.066 |
| Random forest | 0.967 | 0.887 | **0.876** | 0.531 | 0.067 |
| LightGBM | 0.996 | 0.906 | **0.897** | 0.577 | 0.063 |

Test PR-AUC chance level = test death rate (0.103).

**4.2 Threshold:** 0.066 (chosen on training, out-of-fold). On the test hospitals: recall **0.82**, precision **0.33** (1 alert in 3 is a death), **25.7%** of the patients alerted, accuracy 0.81.

**4.3 Calibration (new check)**

| | Observed mortality | Mean predicted, LightGBM | Mean predicted, logistic regression |
|---|---|---|---|
| Training hospitals (out-of-fold) | 0.0813 | 0.0703 | 0.0807 |
| Test hospitals | 0.1026 | 0.0898 | 0.0994 |

> To discuss: the logistic regression follows the higher death rate of the test hospitals (0.099 vs 0.103): the extra deaths are explained by sicker patients, not by a "hospital effect". LightGBM under-predicts by about 13% **in both** training and test hospitals, so its problem is not the new hospitals but its own over-confidence (63 leaves, training AUC 0.999: link with Sections 3.7 and 3.9). On the calibration plot, the LightGBM points are above the diagonal.

**4.4 Important features:** LightGBM (gain): `d1_lactate_min`, `ventilated_apache`, `gcs_motor_apache`, `d1_lactate_max`, `age`, `d1_spo2_min`, `d1_bun_min`... Logistic regression (largest coefficients): `age` (+), `ventilated_apache` (+), `gcs_motor_apache` (−), `d1_heartrate_max` (+), `d1_lactate_min` (+), `d1_spo2_min` (−). Both agree and are clinically sensible.

**4.5 vs APACHE IVa (test patients with a score):** ROC-AUC **0.898 vs 0.856**; to reach ≈ 80% recall, APACHE alerts **33.6%** of patients, LightGBM **25.1%**. On the calibration plot, APACHE lies below the diagonal (over-estimates the risk).

**4.6 Errors, hospitals, subgroups**
- Missed deaths vs detected deaths: ventilated 38% vs 76%, motor GCS 5.6 vs 4.0, max lactate 1.9 vs 5.8, missing values 58.8 vs 50.7, elective surgery 18% vs 5%. They looked stable on day 1.
- Per hospital (23 hospitals with ≥ 10 deaths): ROC-AUC from **0.81 to 0.955** (median 0.904); mean predicted risk 0.048 to 0.110 vs observed 0.045 to 0.159.
- Subgroups: ROC-AUC falls with age (**0.939** < 45 → **0.843** 80+); recall is lowest for **elective surgery (0.55)** and patients **under 45 (0.76)**, where the death rate is low; F and M have the same ROC-AUC (0.897); recall 0.86 (F) vs 0.79 (M).

---

## Part 4. Still to do

1. Write the observations (all the "…").
2. Fill the "…" in `docs/presentation_notes.md` from Part 3, and update the slides (the deck on Marina's Desktop still has the old numbers and structure).
3. Read the whole notebook together: each of you must be able to explain every choice in Part 2 at the board.
4. Commit and push when you're happy.
