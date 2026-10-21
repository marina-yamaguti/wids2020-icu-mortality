# Presentation notes — ICU Mortality Prediction (WiDS 2020)

10 minutes, 10 slides, 3 speakers (about 3 min 20 s each). Numbers come from `newproject.ipynb`.
**"…" = number to fill in after the next full run of the notebook.** **[Fig: x.y]** = figure to export from that section.

| Speaker | Slides | Topic |
|:--|:--|:--|
| Fatoumata | 1–3 | Task, data, state of the art, our angle |
| Lydia | 4–6 | Preprocessing, validation, models |
| Marina | 7–10 | Overfitting, test results vs APACHE, conclusion |

---

## Fatoumata

### Slide 1 — Title and task (≈ 1 min)
- *ICU Mortality Prediction* — the three names, the course.
- **Task:** from the first 24 h of an ICU stay, predict whether the patient dies before leaving the hospital (`hospital_death`).
- Binary classification on tabular data. The target is only known at discharge; the features are routine measurements.
- **Data:** WiDS Datathon 2020 — **91,713 ICU stays, 186 columns, 147 hospitals**.

### Slide 2 — Metrics (≈ 0:50)
- Only **8.63%** of the patients die → "always survives" = **91.37% accuracy** and detects nobody → accuracy is useless.
- **ROC-AUC** (Kaggle metric: probability that a death is ranked above a survivor), **PR-AUC**, **recall / precision** at a threshold, **calibration**.

### Slide 3 — State of the art and teacher's feedback (≈ 1:30)
- 951 Kaggle teams. Winner **0.915**; the top 50 is within **0.005**. Recipe: huge ensembles (50+ models).
- Only 9 of the 66 hospitals of Kaggle's test set are in the training data → **new hospitals**.
- **Teacher's feedback** → our three questions: overfitting of the training hospitals? other sources of overfitting? how close can simple models get?

---

## Lydia

### Slide 4 — Exploration and cleaning (≈ 1:10) **[Fig: 2.3 missing values]**
- **99.97%** of the rows have a missing value; 74 columns are more than 50% missing.
- **Missingness is informative:** the patients with the fewest missing values have the highest mortality (doctors test more when worried).
- **No single feature solves the task:** the best measured feature (max lactate) reaches an AUC of only **…**.
- Cleaning: inverted min/max swapped, negative stays and APACHE `-1` → missing, diagnosis codes → categories.

### Slide 5 — Leakage and validation (≈ 1:00)
- Identifiers removed; **APACHE predictions** = another model's output → removed from the features, kept as a benchmark.
- **Split by hospital** (`GroupShuffleSplit`): the test set = 20% of the hospitals, never used before the end.
- All choices with **`GroupKFold`** (5 folds, grouped by hospital) and `cross_validate(..., return_train_score=True)` → train vs validation score.

### Slide 6 — Models (≈ 1:10) **[Fig: 3.5 num_leaves bar chart + 3.6 boxplot]**
- Dummy 0.5 · Logistic regression **…** · Random forest **…** · LightGBM **…** (on new hospitals).
- Grid searches: `C` for the logistic regression, `min_samples_leaf` for the random forest, `num_leaves` for LightGBM.
- Fold-to-fold std ≈ **…** — about the spread of the whole Kaggle top 50.

---

## Marina

### Slide 7 — Overfitting: patients, hospitals, complexity (≈ 1:10) **[Fig: 3.9 validation curve]**
- **Training patients:** train vs validation gap: logistic regression **…**, random forest **…**, LightGBM **…** → the more flexible, the more it memorizes.
- **Training hospitals:** `hospital_id` helps with a random split (**…**) but not with a grouped split (**…**) → it learns hospital death rates that don't transfer.
- **Complexity:** with more trees the training score → 1 while the validation score stops improving after **…** trees.

### Slide 8 — Hidden overfitting (≈ 1:00)
- **Hospital fingerprints:** the hospital is recognized **…** of the time from the features (chance **…**), **…** from the missing pattern alone.
- **Leaky preprocessing:** target encoding fitted on all the data → **fake …** (above the Kaggle winner!); inside a pipeline → **…**.
- Lesson from the course: fit every preprocessing step on the training folds only (like the PCA in the TP).

### Slide 9 — Test results and APACHE (≈ 1:10) **[Fig: 4.2 confusion matrix + 4.5 calibration]**
- Test hospitals: LightGBM ROC-AUC **…**, PR-AUC **…**.
- Threshold for **80% recall** chosen on the training hospitals → recall **…** on the test hospitals, **1 alert in …** is a real death, **…** of the patients alerted.
- **vs APACHE IVa:** **…** vs **…**; to catch 80% of the deaths, APACHE alerts **…** of the patients, our model **…**.

### Slide 10 — Limits and conclusion (≈ 0:50)
- Missed deaths looked stable on day 1 (less ventilated, better Glasgow, lower lactate).
- Per hospital, the AUC goes from **…** to **…** → a local check is needed before use in a new hospital.
- **Take-away:** one simple LightGBM, tuned on new hospitals, gets within **…** of the best single Kaggle model and beats APACHE; the main risks are overfitting patients, hospitals and leaky preprocessing.

---

## Likely questions (everyone should know these)

| Question | Answer |
|:--|:--|
| Why not use the winners' ensemble? | The gain is small (about one fold std) for 50 models; the teacher asked for simple models; ours is checked on new hospitals. |
| Why is your score lower than Kaggle? | Different test patients (Kaggle labels are private), and ours are entirely unseen hospitals. Not directly comparable. |
| What is APACHE? | A standard ICU severity score: a fixed formula of the worst first-day values, age, diagnosis and chronic conditions. We use it as a benchmark. |
| Why not use APACHE as a feature? | It's another model's prediction, and it didn't help. |
| Why `GroupKFold` and not `train_test_split`? | A new hospital is the real use case; with a random split, the model can use hospital-specific information that doesn't transfer (Section 3.8). |
| What does `add_indicator=True` do? | It adds a 0/1 column "this value was missing" for each feature, because missingness itself predicts death. |
| Why 80% recall? | Screening tool: missing a death is worse than an extra review. It's a clinical choice, one parameter in the code. |
| Why did the naive target encoding score so high? | Each validation patient's own outcome was inside its group's death rate: the feature contained the answer. |
| Why is the accuracy lower than 91%? | We accept false alarms to catch 80% of the deaths; accuracy rewards "always survives". |
