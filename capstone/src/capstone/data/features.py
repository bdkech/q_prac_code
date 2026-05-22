import polars as pl
from loguru import logger

# Ordered candidate feature set for the PI-CAI QSVM.
# psa_missing, gleason_primary_max, and primary_pattern_4plus require
# upstream derivation steps not yet wired in; build_feature_matrix warns
# when they are absent rather than crashing the pipeline.
CORE_FEATURES: list[str] = [
    "psa",
    "psad",
    "psa_missing",
    "prostate_volume",
    "grade_group_max",
    "gleason_primary_max",
    "primary_pattern_4plus",
    "patient_age",
    "n_mri_studies",
    "had_multiple_mri",
]


def derive_psa_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Derive PSA density fallback and missing-value flag from available columns.

    PI-CAI reports psad directly in ~60–70% of studies. When it is absent,
    it can be approximated as psa / prostate_volume (the standard clinical
    formula). The psa_missing flag preserves the information that the density
    was not directly measured — absence of a recorded value is itself a weak
    clinical signal.

    This function must be called on the patient-level DataFrame (after
    aggregate_to_patient) so that psa and prostate_volume already reflect
    the most-recent-study values.

    Args:
        df: Patient-level DataFrame containing columns psa, psad, and
            prostate_volume.

    Returns:
        DataFrame with psad filled where derivable and a new boolean column
        psa_missing (True when psad is still null after the fallback).

    Raises:
        TypeError: If df is not a polars DataFrame.
        ValueError: If psa, psad, or prostate_volume columns are absent.
    """
    if not isinstance(df, pl.DataFrame):
        error_msg = f"Input must be polars DataFrame, got {type(df).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    required = ["psa", "psad", "prostate_volume"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    n_null_before = df["psad"].null_count()

    # Fill psad from psa / prostate_volume when both source columns are present
    df = df.with_columns(
        pl.when(
            pl.col("psad").is_null()
            & pl.col("psa").is_not_null()
            & pl.col("prostate_volume").is_not_null()
        )
        .then(pl.col("psa") / pl.col("prostate_volume"))
        .otherwise(pl.col("psad"))
        .alias("psad")
    )

    n_derived = n_null_before - df["psad"].null_count()
    n_still_missing = df["psad"].null_count()

    logger.debug(
        f"PSA density: {n_derived} values derived from psa/prostate_volume, "
        f"{n_still_missing} still null"
    )

    # Flag rows where psad remains null — imputation happens downstream
    df = df.with_columns(pl.col("psad").is_null().alias("psa_missing"))

    return df


def build_feature_matrix(
    df: pl.DataFrame,
    feature_cols: list[str] | None = None,
    label_col: str = "label_cspc",
) -> tuple[pl.DataFrame, pl.Series]:
    """
    Separate the feature matrix X and label vector y from a patient DataFrame.

    Logs a warning for any CORE_FEATURES columns that are absent so callers
    can detect missing upstream derivation steps without crashing. Only the
    columns that are actually present in df are included in X — this lets the
    pipeline run incrementally as new feature steps are added.

    Args:
        df: Patient-level DataFrame from aggregate_to_patient and any
            upstream feature derivation steps.
        feature_cols: Ordered feature column names. Defaults to CORE_FEATURES.
        label_col: Binary classification label column (0 = no csPCa, 1 = csPCa).

    Returns:
        Tuple (X, y):
        - X: DataFrame with only the available requested feature columns.
        - y: Integer Series of labels.

    Raises:
        TypeError: If df is not a polars DataFrame.
        ValueError: If label_col is not present in df.
    """
    if not isinstance(df, pl.DataFrame):
        error_msg = f"Input must be polars DataFrame, got {type(df).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if label_col not in df.columns:
        error_msg = f"Label column '{label_col}' not found in DataFrame"
        logger.error(error_msg)
        raise ValueError(error_msg)

    cols = feature_cols if feature_cols is not None else CORE_FEATURES

    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(
            f"Feature columns absent (upstream steps may not have run): {missing}"
        )

    available = [c for c in cols if c in df.columns]
    logger.debug(
        f"Feature matrix: {len(available)}/{len(cols)} features, {len(df)} patients"
    )

    return df.select(available), df[label_col]
