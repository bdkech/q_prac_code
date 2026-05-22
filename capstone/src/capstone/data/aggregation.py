import polars as pl
from loguru import logger


def aggregate_to_patient(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate MRI study data to one row per patient.

    Groups by patient_id and sorts studies chronologically by mri_date.
    Takes demographic/clinical data from the most recent study and derives
    aggregate features across all studies for each patient.

    PI-CAI is structured per-study, not per-patient. ~24 of 1,476 patients
    have more than one study; aggregation is required for correctness even
    though it is a no-op for 98% of the cohort.

    Args:
        df: DataFrame with columns:
            patient_id, mri_date, patient_age, psa, psad, prostate_volume,
            center, is_cspc (bool), grade_group_max (int).

    Returns:
        DataFrame with one row per patient containing:
        patient_id, patient_age, psa, psad, prostate_volume, center,
        n_mri_studies, had_multiple_mri (bool), label_cspc (int 0/1),
        grade_group_max.

    Raises:
        TypeError: If df is not a polars DataFrame.
        ValueError: If any required column is absent from df.
    """
    if not isinstance(df, pl.DataFrame):
        error_msg = f"Input must be polars DataFrame, got {type(df).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    required_cols = [
        "patient_id",
        "mri_date",
        "patient_age",
        "psa",
        "psad",
        "prostate_volume",
        "center",
        "is_cspc",
        "grade_group_max",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    n_studies = len(df)
    n_unique_patients = df["patient_id"].n_unique()
    logger.debug(f"Aggregating {n_studies} studies from {n_unique_patients} patients")

    # Most recent study (last in chronological order) provides clinical values
    df_sorted = df.sort(["patient_id", "mri_date"])

    patient_df = df_sorted.group_by("patient_id").agg(
        [
            pl.len().alias("n_mri_studies"),
            pl.col("patient_age").last().alias("patient_age"),
            pl.col("psa").last().alias("psa"),
            pl.col("psad").last().alias("psad"),
            pl.col("prostate_volume").last().alias("prostate_volume"),
            pl.col("center").last().alias("center"),
            # Worst-case pathology across all studies
            pl.col("is_cspc").any().alias("any_cspc"),
            pl.col("grade_group_max").max().alias("grade_group_max"),
        ]
    )

    patient_df = patient_df.with_columns(
        [
            (pl.col("n_mri_studies") > 1).alias("had_multiple_mri"),
            # Primary classification target: 1 if any study showed csPCa
            pl.col("any_cspc").cast(pl.Int32).alias("label_cspc"),
        ]
    ).drop("any_cspc")

    patient_df = patient_df.sort("patient_id")

    mean_studies = patient_df["n_mri_studies"].mean()
    n_with_multiple = patient_df["had_multiple_mri"].sum()
    n_with_cspc = patient_df["label_cspc"].sum()

    logger.debug(
        f"Aggregated to {len(patient_df)} patients. "
        f"Mean studies per patient: {mean_studies:.2f}"
    )
    logger.debug(
        f"Patients with multiple MRIs: {n_with_multiple} "
        f"({100 * n_with_multiple / len(patient_df):.1f}%)"
    )
    logger.debug(
        f"Patients with csPC: {n_with_cspc} "
        f"({100 * n_with_cspc / len(patient_df):.1f}%)"
    )

    return patient_df
