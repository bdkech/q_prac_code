import polars as pl
from loguru import logger


def fit_scaler(
    train_df: pl.DataFrame,
    feature_cols: list[str],
) -> dict[str, tuple[float, float]]:
    """
    Compute per-column mean and standard deviation from training data only.

    Fitting exclusively on training data and reusing the same statistics for
    val/test prevents leakage from future observations. Polars mean/std
    naturally skip nulls, so imputation must happen upstream for any columns
    with missing values before calling this function.

    Args:
        train_df: Training-split DataFrame with the feature columns.
        feature_cols: Columns to compute statistics for. Must all exist.

    Returns:
        Dict mapping each column name to a (mean, std) tuple. If a column
        has zero variance, std is stored as 1.0 — apply_scaler will
        correctly produce all-zeros for that column after centering.

    Raises:
        TypeError: If train_df is not a polars DataFrame.
        ValueError: If any column in feature_cols is absent from train_df.
    """
    if not isinstance(train_df, pl.DataFrame):
        error_msg = f"Input must be polars DataFrame, got {type(train_df).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        error_msg = f"Columns not found in training DataFrame: {missing}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    scaler: dict[str, tuple[float, float]] = {}

    for col in feature_cols:
        series = train_df[col].cast(pl.Float64)
        mean: float = series.mean() or 0.0
        std: float = series.std() or 0.0

        if std == 0.0:
            # Zero-variance features are all-zero after centering;
            # setting std=1.0 avoids divide-by-zero in apply_scaler.
            logger.warning(
                f"Column '{col}' has zero variance in training set — "
                f"will be zeroed out after centering"
            )
            std = 1.0

        scaler[col] = (mean, std)

    logger.debug(f"Fitted scaler for {len(scaler)} columns")
    return scaler


def apply_scaler(
    df: pl.DataFrame,
    scaler: dict[str, tuple[float, float]],
) -> pl.DataFrame:
    """
    Apply zero-mean / unit-variance scaling using a pre-fitted scaler.

    Only columns present in both the scaler dict and df are transformed.
    Non-feature columns (e.g., patient_id) are passed through untouched,
    so the full DataFrame does not need to be stripped before calling.

    Args:
        df: DataFrame to transform. May contain non-feature columns.
        scaler: Column → (mean, std) mapping from fit_scaler.

    Returns:
        DataFrame with scaled feature columns cast to Float64. Column
        order matches the input DataFrame.

    Raises:
        TypeError: If df is not a polars DataFrame or scaler is not a dict.
    """
    if not isinstance(df, pl.DataFrame):
        error_msg = f"Input must be polars DataFrame, got {type(df).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    if not isinstance(scaler, dict):
        error_msg = f"Scaler must be a dict, got {type(scaler).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    skipped = [c for c in scaler if c not in df.columns]
    if skipped:
        logger.warning(f"Scaler columns absent from DataFrame (skipped): {skipped}")

    scale_exprs = [
        ((pl.col(col).cast(pl.Float64) - mean) / std).alias(col)
        for col, (mean, std) in scaler.items()
        if col in df.columns
    ]

    if not scale_exprs:
        logger.warning("No columns to scale — returning DataFrame unchanged")
        return df

    logger.debug(f"Scaling {len(scale_exprs)} columns")
    return df.with_columns(scale_exprs)
