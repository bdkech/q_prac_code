# Prostate cancer QSVM — PI-CAI dataset implementation guide

**Project:** Quantum practitioners course capstone  
**Dataset:** PI-CAI: Public Training and Development Dataset (v2.0, 2025)  
**Source:** `github.com/DIAGNijmegen/picai_labels` + `zenodo.org/records/6624726`  
**Hardware:** IBM Heron r2 (156 qubits, heavy-hex topology)  
**Model:** Quantum support vector machine (QSVM) with fidelity quantum kernel  
**Last updated:** May 2026

This document is the PI-CAI-specific implementation companion to `prostate_qsvm_data_decisions.md`. It maps the abstract feature engineering decisions in that document onto the concrete columns and structure of the PI-CAI `marksheet.csv`, notes where the dataset covers the design and where it falls short, and records any PI-CAI-specific processing steps needed before the general pipeline applies.

---

## 1. Dataset overview

PI-CAI contains 1,500 biparametric MRI (bpMRI) studies from 1,476 patients acquired 2012–2021 at three Dutch centers (Radboud UMC, UMC Groningen, Ziekenhuis Groep Twente). It is fully anonymized and freely downloadable from Zenodo — no data transfer agreement required.

**Important structural note:** PI-CAI is organized by MRI study, not by patient. The dataset contains 1,500 studies from 1,476 patients — a difference of only 24, meaning repeat studies affect roughly 1.6% of the cohort. PI-CAI is effectively a single-timepoint dataset; the rare multi-study patients are the exception, not a structural feature of the data. Per-patient aggregation is still required for correctness, but features derived from change over time (`pirads_delta`, `had_multiple_mri`) will be near-zero variance and are expected to be dropped by L1 selection.

### Key files

| File | Contents |
|---|---|
| `clinical_information/marksheet.csv` | One row per MRI study. Contains all tabular clinical and outcome variables. This is the primary working file. |
| `csPCa_lesion_delineations/` | NIfTI segmentation masks — not needed for tabular QSVM. |
| `anatomical_delineations/` | Prostate zone segmentations — not needed for tabular QSVM. |

---

## 2. `marksheet.csv` column inventory

The columns used for this project, with their PI-CAI names and notes:

| PI-CAI column | Type | Maps to design doc feature | Notes |
|---|---|---|---|
| `patient_id` | string | patient identifier | Multiple rows possible but affects only ~24 of 1,476 patients (~1.6%) |
| `study_id` | string | study identifier | One row per MRI study |
| `patient_age` | float | demographic covariate | Age at time of MRI exam |
| `PSA` / `PSA_report` | float (ng/mL) | `psa_baseline`, `psa_peak` | Single value per exam, not longitudinal |
| `PSAD_report` | float (ng/mL²) | `psa_density` | Directly reported; use over derived value where available |
| `prostate_volume_report` | float (mL) | prostate volume for density check | Clinical estimate using prolate ellipsoid model |
| `prostate_volume_automatic` | float (mL) | prostate volume backup | From auto-segmentation; use as fallback if `prostate_volume_report` missing |
| `PIRADS` | integer (1–5) | `radiology_composite_max` | Overall PI-RADS score as reported; see section 4.2 for zone-aware derivation caveat |
| `histology_type` | categorical | biopsy type flag | `SysBx`, `MRBx`, `SysBx+MRBx`, `RP`, or missing (negative study) |
| `GS` | string | grade group / Gleason label | Comma-separated per lesion if multiple; see section 3 for parsing |
| `scanner_manufacturer` | categorical | site/scanner covariate | Siemens, Philips, or GE — consider as batch covariate |
| `scanner_model_name` | categorical | scanner covariate | For batch effect check |

**Columns not present in PI-CAI** (gaps vs. design doc):

| Design doc feature | Status in PI-CAI |
|---|---|
| `psa_velocity`, `psa_doubling_time`, `psa_cv` | **Absent** — only one PSA value per exam, no longitudinal draws |
| `radiology_composite` (zone-aware T2/DWI/DCE components) | **Absent** — PI-CAI is biparametric (T2 + DWI only); no DCE; component scores not tabulated |
| `dce_positive` | **Absent** — DCE excluded by design in bpMRI |
| `zone_peripheral`, `zone_transition` | **Partially available** — zone derives from segmentation masks, not a marksheet column; approximated from PIRADS zone context if noted in GS string |
| `prior_negative_biopsy`, `surveillance_duration_days` | **Absent** — single-exam dataset; no longitudinal visit history |
| `time_psa_to_mri_days`, `time_mri_to_biopsy_days` | **Absent** — no pre-MRI timeline data |

The PSA trajectory features and pathway structure features from the design doc are largely unavailable in PI-CAI. The radiological composite feature group is partially available via the reported PIRADS score but not reconstructable at the component level. This is expected — PI-CAI is a single-timepoint MRI dataset, not a longitudinal surveillance registry.

---

## 3. Label derivation

### Parsing the `GS` column

The `GS` column contains comma-separated Gleason scores in `primary+secondary` format (e.g., `3+4,3+3`), one entry per lesion sampled. For patients with no tissue sampling (`histology_type` missing), `GS` is also missing — these are negative studies.

Parse to extract:

```python
import pandas as pd

def parse_gs(gs_string):
    """
    Returns (grade_group_max, gleason_sum_max, gleason_primary_max, is_cspc)
    for a single GS cell. Returns None tuple if no biopsy.
    """
    if pd.isna(gs_string):
        return None, None, None, False
    
    scores = [s.strip() for s in gs_string.split(',')]
    grade_groups = []
    sums = []
    primaries = []
    
    for s in scores:
        if '+' not in s:
            continue
        parts = s.split('+')
        primary, secondary = int(parts[0]), int(parts[1])
        gs_sum = primary + secondary
        gg = gleason_to_grade_group(primary, secondary)
        grade_groups.append(gg)
        sums.append(gs_sum)
        primaries.append(primary)
    
    if not grade_groups:
        return None, None, None, False
    
    gg_max = max(grade_groups)
    return gg_max, max(sums), max(primaries), gg_max >= 2

def gleason_to_grade_group(primary, secondary):
    gs = primary + secondary
    if gs <= 6:
        return 1
    elif gs == 7 and primary == 3:
        return 2
    elif gs == 7 and primary == 4:
        return 3
    elif gs == 8:
        return 4
    else:  # 9 or 10
        return 5
```

### Cohort filter

Apply the following filters before building any features:

1. **Keep only rows where `histology_type` is not missing** — studies with no biopsy have no ground-truth label and must be excluded from training/evaluation.
2. **Exclude `histology_type == 'RP'`** — radical prostatectomy specimens are post-treatment; their pathology cannot be treated as a biopsy label without risk of post-event leakage (see section 6).
3. After parsing `GS`, **drop rows where `grade_group_max` is null** — this catches any malformed GS strings.

**Resulting label:**
- **Positive (csPCa):** `grade_group_max >= 2`
- **Negative:** `grade_group_max == 1` (confirmed insignificant disease at biopsy)

### Expected class balance

Of the 1,500 studies, 425 are annotated as csPCa (~28%), giving a positive rate consistent with the 30–50% range noted in the design doc. Class weighting is unlikely to be required, but verify after applying the cohort filter above, as the filtered biopsied subset may differ.

---

## 4. Feature engineering on PI-CAI

### 4.1 Per-patient aggregation

Since `marksheet.csv` is study-indexed, aggregate to one row per `patient_id` before feature construction. In practice this step affects only ~24 patients (1.6% of the cohort) — the vast majority already have exactly one study. The aggregation is still required for correctness, but do not treat it as a meaningful data transformation.

For the ~24 patients with multiple studies:

- Use the **most recent study** as the primary record for PSA and PIRADS values (closest to biopsy decision).
- Count of studies becomes `n_mri_studies`.
- Record whether `PIRADS` changed between first and most recent study as `pirads_delta`.

Both `n_mri_studies` and `pirads_delta` will have near-zero variance across the full cohort and are expected to be eliminated at the L1 feature selection step. Include them in the candidate set for completeness but do not rely on them as informative features.

```python
def aggregate_patient(group):
    group = group.sort_values('study_id')  # studies are chronologically ordered by ID
    most_recent = group.iloc[-1]
    return pd.Series({
        'patient_age':            most_recent['patient_age'],
        'psa':                    most_recent['PSA'],
        'psa_density':            most_recent['PSAD_report'],
        'prostate_volume':        most_recent['prostate_volume_report'].fillna(
                                      most_recent['prostate_volume_automatic']),
        'pirads_max':             group['PIRADS'].max(),
        'pirads_delta':           group['PIRADS'].iloc[-1] - group['PIRADS'].iloc[0]
                                  if len(group) > 1 else 0,
        'n_mri_studies':          len(group),
        'had_multiple_mri':       int(len(group) > 1),
        'scanner_manufacturer':   most_recent['scanner_manufacturer'],
        # label fields — from the biopsied study
        'grade_group_max':        group['grade_group_max'].max(),
        'gleason_sum_max':        group['gleason_sum_max'].max(),
        'gleason_primary_max':    group['gleason_primary_max'].max(),
        'label_cspc':             int(group['is_cspc'].any()),
    })

patient_df = df_filtered.groupby('patient_id').apply(aggregate_patient).reset_index()
```

### 4.2 PSA features

PI-CAI provides a single PSA value per exam, not repeated longitudinal draws. The trajectory features (`psa_velocity`, `psa_doubling_time`, `psa_cv`) from the design doc are therefore unavailable. The available PSA-derived features are:

| Feature | Derivation in PI-CAI | Notes |
|---|---|---|
| `psa` | `PSA` from most recent study | Treat as `psa_baseline` proxy |
| `psa_density` | `PSAD_report` directly; or `PSA / prostate_volume` if missing | Use reported value first — clinical rounding means derived ≠ reported |
| `psa_missing` | Binary flag: `PSAD_report` is null | Missing PSA density is informative (not reported = not concerning enough to document) |

**Gap acknowledgement:** The absence of longitudinal PSA is the most significant feature gap between PI-CAI and the real-world dataset design. In the course capstone context, `psa_density` alone is the primary PSA signal. The trajectory features will be recoverable when moving to the real-world dataset.

### 4.3 Radiological features

PI-CAI is **biparametric** (T2-weighted + DWI with ADC map). DCE is excluded by design. The component-level T2/DWI/DCE scores needed to reconstruct the zone-aware composite from the design doc are not tabulated in `marksheet.csv` — only the overall `PIRADS` score is reported.

Use `PIRADS` directly as the radiological feature, with the following derived columns:

| Feature | Derivation |
|---|---|
| `pirads_max` | Highest PIRADS score across all studies (per patient aggregation above) |
| `pirads_delta` | Change in PIRADS from first to most recent study (0 for single-study patients; near-zero variance — expect L1 to drop) |
| `pirads_high` | Binary: `pirads_max >= 4` (threshold for strong suspicion) |
| `n_mri_studies` | Count of studies per patient; near-zero variance — expect L1 to drop |
| `had_multiple_mri` | Binary: more than one MRI study; near-zero variance — expect L1 to drop |

**Note on zone-aware composite:** The zone-aware composite formula in the design doc cannot be applied to PI-CAI because component scores (T2, DWI, DCE separately) are not tabulated. The reported `PIRADS` score already encodes PI-RADS v2.1 zone-weighting, so it serves as a reasonable proxy. When moving to the real-world dataset, revisit whether component scores are available separately to reconstruct the composite.

### 4.4 Pathology features

Derived from the parsed `GS` column after per-patient aggregation:

| Feature | Derivation |
|---|---|
| `grade_group_max` | Highest grade group across all biopsies for this patient |
| `gleason_primary_max` | Highest primary Gleason pattern recorded |
| `gleason_sum_max` | Highest Gleason sum recorded |
| `primary_pattern_4plus` | Binary: any primary Gleason pattern ≥ 4 |

**Leakage check:** In PI-CAI, `histology_type == 'RP'` rows represent radical prostatectomy specimens taken after treatment. These must be excluded from the cohort (step 3 of cohort filter above) — their pathology is definitionally post-event. Biopsy rows (`SysBx`, `MRBx`, `SysBx+MRBx`) are the valid label source.

### 4.5 Lesion localization

Zone (peripheral vs. transition) is not a direct column in `marksheet.csv`. It is encoded in the segmentation masks (`anatomical_delineations/`), which requires loading NIfTI files. For the tabular QSVM this is not worth the processing overhead. Options:

- **Skip zone features** for PI-CAI capstone — the PIRADS score already partially encodes zone (PI-RADS 4 in TZ vs. PZ has different implications, and the radiologist applies this when assigning the score).
- **Add a single proxy:** `pirads_high` (≥ 4) captures the zone-adjusted clinical threshold implicitly.
- **Revisit for real-world data** where zone is likely a direct column in the EMR.

### 4.6 Structural / pathway features available in PI-CAI

| Feature | Availability | Notes |
|---|---|---|
| `n_mri_studies` | ✓ Available | Near-zero variance (~98% of patients = 1); expect L1 to drop |
| `had_multiple_mri` | ✓ Available | Near-zero variance (~98% of patients = 0); expect L1 to drop |
| `scanner_manufacturer` | ✓ Available | Useful as a batch/site covariate for variance analysis; not a clinical feature — exclude from QSVM feature vector but track for confound checks |

Features absent from PI-CAI: `prior_negative_biopsy`, `surveillance_duration_days`, `time_mri_to_biopsy_days`, `n_psa_draws`. These are all longitudinal pathway features that require a surveillance registry structure.

---

## 5. Revised candidate feature set for PI-CAI

Applying the PI-CAI constraints to the design doc's 21-candidate list gives the following reduced but coherent set:

| Group | Feature | Source column(s) | Notes |
|---|---|---|---|
| PSA | `psa` | `PSA` (most recent study) | Single-timepoint proxy for baseline |
| PSA | `psa_density` | `PSAD_report` → fallback `PSA / prostate_volume` | Strongest single PSA feature |
| PSA | `psa_missing` | Binary flag on `PSAD_report` nullness | Structural missingness signal |
| Radiological | `pirads_max` | `PIRADS` (max across studies; = single value for 98% of patients) | Primary radiological feature |
| Radiological | `pirads_high` | Binary: `pirads_max >= 4` | Zone-adjusted clinical threshold |
| Pathology | `grade_group_max` | Parsed from `GS` | Highest ISUP grade recorded |
| Pathology | `gleason_primary_max` | Parsed from `GS` | Primary pattern ordering |
| Pathology | `primary_pattern_4plus` | Binary derived from `GS` | High-risk threshold flag |
| Demographic | `patient_age` | `patient_age` | Age at MRI exam |

**Total: 9 core candidate features.** This is slightly below the 10–14 target from the design doc, reflecting PI-CAI's single-timepoint structure. The near-zero-variance structural features (`n_mri_studies`, `pirads_delta`, `had_multiple_mri`) are omitted here — include them in the initial L1 run to confirm they drop, then remove before PCA.

The reduced candidate count is not a problem for the capstone: 9 features feeding PCA to 6–8 components is well within the ZZFeatureMap qubit budget, and the L1 step may preserve all 9 if they each carry independent variance. The feature gap relative to the design doc is expected and documented in section 9 for real-world data recovery.

---

## 6. Temporal leakage in PI-CAI context

PI-CAI's single-timepoint structure simplifies the temporal leakage problem relative to the longitudinal design: there is no timeline to traverse per patient. However, two leakage risks remain specific to this dataset:

**Risk 1 — RP pathology rows.** Rows with `histology_type == 'RP'` contain post-prostatectomy pathology. If a patient has both a biopsy row and an RP row in the data, and the RP row has a higher grade group, using `grade_group_max` across all rows would incorporate post-event pathology. The cohort filter in section 3 (exclude RP rows before aggregation) prevents this.

**Risk 2 — PIRADS assigned after biopsy decision.** In a small number of cases, an MRI may have been acquired after a biopsy was already planned or performed. Since PI-CAI does not provide absolute dates (only anonymized relative study identifiers), this cannot be verified directly. Treat this as an acceptable residual risk for the capstone; note it as a limitation.

---

## 7. Data split and preprocessing

Follow the design doc recommendations directly:

- **Stratified 70 / 15 / 15 train / validation / test split** on `patient_id` (not `study_id`) to prevent patient leakage across splits.
- Standardize all continuous features (zero mean, unit variance) fit on training set only.
- Apply PCA on standardized training features; transform val and test using training PCA parameters.
- Target **6–8 principal components** capturing ≥ 90% of variance for ZZFeatureMap encoding.

PI-CAI provides pre-computed 5-fold cross-validation splits in `picai_baseline` (GitHub: `DIAGNijmegen/picai_baseline`) that guarantee no patient overlap. These can be used as-is or as a reference for constructing the 70/15/15 split.

### Handling missingness before standardization

Several columns have real-world missingness:

| Column | Missing rate (approximate) | Strategy |
|---|---|---|
| `PSAD_report` | ~30–40% | Derive from `PSA / prostate_volume_report`; if both missing, impute median **from training set only** and set `psa_missing = 1` |
| `prostate_volume_report` | ~15–20% | Fall back to `prostate_volume_automatic` (auto-segmentation) before declaring missing |
| `PIRADS` | Rare | Drop row — PIRADS is the core radiological feature and a study without it is likely a data quality issue |

Do not apply KNN or multivariate imputation — the sample size after filtering (~400–600 biopsied patients) is too small for reliable multivariate imputation, and the QSVM kernel is sensitive to imputed data points as noted in the design doc.

---

## 8. Site/scanner batch effects

PI-CAI spans three Dutch centers with different MRI vendors (Siemens, Philips, GE). PIRADS scores and prostate volume estimates may carry systematic inter-site differences. Before finalizing the feature set:

1. Compute per-feature distributions stratified by `scanner_manufacturer`.
2. If `pirads_max` or `psa_density` distributions differ significantly across manufacturers (KS test or visual inspection), add `scanner_manufacturer` as a one-hot covariate in the L1 feature selection step — not necessarily in the final QSVM feature vector, but to check whether it absorbs variance that would otherwise confound the kernel.
3. If site effects are substantial, consider training on two centers and holding out the third as an out-of-distribution test.

---

## 9. Feature gaps vs. real-world dataset — transition notes

When moving from PI-CAI to the real-world dataset post-course, the following features from the design doc become recoverable:

| Design doc feature | PI-CAI status | Real-world recovery |
|---|---|---|
| `psa_velocity` | Unavailable | Direct: multiple PSA draws per patient in EMR |
| `psa_doubling_time` | Unavailable | Direct: requires ≥ 3 draws |
| `psa_cv` | Unavailable | Direct: coefficient of variation across draws |
| `radiology_composite` (zone-aware) | Proxy only (PIRADS) | Direct: T2/DWI/DCE component scores if stored separately |
| `dce_positive` | Unavailable (bpMRI) | Direct: if mpMRI with DCE is standard of care at your site |
| `zone_peripheral`, `zone_transition` | Absent from marksheet | Direct: likely in structured radiology report or lesion table |
| `prior_negative_biopsy` | Absent | Direct: biopsy history table in EMR |
| `surveillance_duration_days` | Absent | Direct: first visit to index biopsy date |
| `time_mri_to_biopsy_days` | Absent | Direct: MRI date + biopsy date |

The 12-feature PI-CAI vector is a proper subset of the 21-candidate real-world vector. The PCA and QSVM pipeline can be ported without structural changes; additional features slot into the pre-PCA candidate set and feed the same L1 selection → PCA → ZZFeatureMap → QSVM chain.

---

## 10. Open questions specific to PI-CAI

- After applying the cohort filter (biopsied patients only, RP excluded), what is the actual N and positive rate? This determines whether 70/15/15 produces a test set large enough to draw meaningful conclusions.
- For patients with multiple MRI studies, does using the most-recent PIRADS value introduce implicit temporal leakage (i.e., was the most recent study performed after the biopsy decision was made)? Check study ID ordering vs. `histology_type` presence.
- Does `prostate_volume_automatic` agree closely enough with `prostate_volume_report` to serve as a fallback, or does the substitution introduce systematic bias in `psa_density`?
- Are any patients represented in both the Zenodo public dataset and the private PI-CAI dataset used in the Lancet Oncology validation paper? This is unlikely to affect the capstone but worth noting if results are compared to published benchmarks.
