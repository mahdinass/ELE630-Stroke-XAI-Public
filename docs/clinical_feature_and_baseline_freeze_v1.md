# ELE630 — Clinical Input Contract and Clinical-Only Reference Freeze v1

**Status:** The 11-feature admission-time clinical predictor set was approved by supervisor Sazidur on 11 October 2026. The fixed clinical-only Stage 5B logistic-regression development baseline was completed and frozen on 12 October. This document records those existing decisions for Strategy V4. **It does not freeze or implement the NCCT+clinical / CTP+clinical fusion mechanism.**

**Record prepared:** 18 October 2026
**Source/provenance:** `src/ele630/clinical_preprocessing.py`, `src/ele630/clinical_baseline.py`, `src/ele630/clinical_runner.py`, `docs/stage5a_clinical_baseline.md`, `docs/stage5b_clinical_runner.md`; verified Stage 5B source revision `46e35fb`; repository review revision `ec886423cc33382553eb2bf9128b1e28a782ce65`.

## 1. Approved predictor schema

The feature order in `FEATURE_NAMES` is fixed and must be identical for the clinical-only reference and both imaging+clinical comparisons:

| Order | Source feature name | Intended meaning |
| ---: | --- | --- |
| 1 | `age` | Age |
| 2 | `sex` | Sex (F=0, M=1) |
| 3 | `nihss_admission` | NIHSS at admission |
| 4 | `mrs_baseline` | Pre-stroke baseline mRS |
| 5 | `onset_to_ct_time` | Recorded onset-to-CT elapsed time |
| 6 | `af_paf` | AF/PAF history or status |
| 7 | `ht` | Hypertension |
| 8 | `dm` | Diabetes |
| 9 | `smoking` | Smoking, encoded as 0 vs any recorded code >0 |
| 10 | `anticoagulation` | Anticoagulation history |
| 11 | `wake_up_stroke` | Wake-up stroke status |

**Prediction landmark:** Strict admission / pre-treatment. The outcome is 90-day mRS (good 0–2, poor 3–6) and must never be a predictor. Discharge NIHSS, treatment results, thrombectomy/recanalization/mTICI, follow-up variables, post-treatment times, raw clock timestamps, patient IDs, acute lesion GT masks and other post-landmark inputs are excluded.

## 2. Frozen clinical preprocessing

Use `ClinicalPreprocessor` as implemented in the cited source. The four numeric variables are age, admission NIHSS, baseline mRS and onset-to-CT duration; impute missing numeric values with the training-partition median, then scale by the training-partition population mean and standard deviation (zero SD uses scale 1). Impute categorical/binary variables using the training-partition mode; ties use the smaller code. Preprocessing is freshly fitted for each development/evaluation fold **only on its training partition** and reused without refitting on that fold's validation/test patients. Do not impute or scale using the pooled 143-patient dataset.

`onset_to_ct_time` is parsed from strings `H:MM` or `H:MM:SS` into minutes. The implemented conservative technical rule masks onset-to-CT duration when `wake_up_stroke == 1` before train-only imputation, even if a source interval is recorded. This is **not** a supervisor-certified clinical interpretation. Preserve the existing implementation as the development reference rather than silently changing it during fusion.

**Scientific provenance limitations to record:** Timing semantics and the source codebook for onset-to-CT (including `H:MM`), wake-up onset ambiguity, exact admission/pre-treatment availability of NIHSS/AF/anticoagulation fields, and possible cohort/protocol shortcut learning are not fully certified by technical schema checks. Supervisor approval of the feature names does not prove case-by-case recording-time eligibility. Disclose these limitations and resolve material evidence gaps before strong clinical claims or final external evaluation; do not silently change features based on reused validation outcomes.

## 3. Frozen clinical-only baseline

| Component | Fixed rule |
| --- | --- |
| Model | scikit-learn `LogisticRegression` |
| Penalty | L2, `C=1.0` |
| Solver | `lbfgs`, `max_iter=1000`, `tol=1e-8` |
| Other configuration | `fit_intercept=True`, `class_weight=None`, `warm_start=False`, `random_state=630`, `verbose=0` |
| Threshold | Fixed poor-outcome probability `>=0.5` for threshold metrics; no tuning |
| Model training | Fold-specific training partition only; no outcome-guided feature search |
| Prior control | Constant poor-outcome prevalence from each training partition |
| Reported runtime | Stage 5B used Python 3.10.11 and scikit-learn 1.2.2 |

Stage 5B fit the model once on 92 fold-0 inner-training patients, then evaluated the frozen configuration on 23 inner-validation patients (16 good, 7 poor). Results from the completed run `stage5b_clinical_dev_01` were privately checked for aggregate integrity. No outer-test clinical predictions were produced in that stage.

| Fold-0 development metric | Clinical-only logistic | Training-prevalence prior |
| --- | ---: | ---: |
| AUROC | **0.9821428571** | 0.5000000000 |
| Average precision | **0.9682539683** | 0.3043478261 |
| Balanced accuracy at 0.5 | **0.9285714286** | 0.5000000000 |
| BCE | **0.2313358984** | 0.6147787322 |
| Brier score | **0.0663344280** | 0.2118383743 |

These are **descriptive 23-case development results**, not final generalization evidence. Very high apparent performance warrants careful assessment of label leakage, feature timing and cohort effects. The current clinical-only baseline is a reference, not a proof of clinical deployability.

## 4. What remains explicitly unfrozen

**Stage 5C3 integration is NOT yet defined, fitted or frozen.** Select and document one suitably low-capacity, patient-level, leakage-safe combination mechanism for the already frozen NCCT and CTP imaging representations with these 11 inputs. Evaluate matched NCCT-only versus NCCT+clinical and CTP-only versus CTP+clinical, plus clinical-only, on the same partitions. Distinguish genuine incremental image information from mere probability shrinkage or improvement caused entirely by clinical inputs. A fixed 50/50 probability average is only a possible control, not required by Strategy V4 or authorized by this document.

Before an outer-fold or external test is opened, freeze the full combined-model fitting, calibration/threshold and evaluation protocols. Every outer fold must train its own appropriate models/preprocessors on authorized training partitions and predict held-out patients exactly once under the predetermined procedure. Clearly disclose repeated fold-0 development exposure and avoid claiming independent unbiased architecture selection from it.

Private clinical rows, coefficients, preprocessing state, patient-level predictions, private paths, images and checkpoints stay on approved UiS storage, **not** in this code repository or shared workspaces. Later changes to the eleven-field schema or baseline configuration require a documented versioned decision and rationale, not a silent adjustment after inspecting held-out outcomes.
