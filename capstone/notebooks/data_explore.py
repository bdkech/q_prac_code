import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    from loguru import logger

    from capstone.data import (
        aggregate_to_patient,
        build_feature_matrix,
        derive_psa_features,
        parse_gs,
    )
    from capstone.preprocessing import apply_scaler, fit_scaler

    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
    )
    return (
        aggregate_to_patient,
        apply_scaler,
        build_feature_matrix,
        derive_psa_features,
        fit_scaler,
        parse_gs,
        pl,
    )


@app.cell
def _(pl):
    my_patient_data = pl.read_csv("data/marksheet.csv")
    return (my_patient_data,)


@app.cell
def _(my_patient_data, parse_gs, pl):
    # Parse lesion_GS into grade_group_max and is_cspc for each study row
    processed_data = my_patient_data.with_columns(
        [
            pl.col("lesion_GS")
            .map_elements(lambda x: parse_gs(x)[0], return_dtype=pl.Int64)
            .alias("grade_group_max"),
            pl.col("lesion_GS")
            .map_elements(lambda x: parse_gs(x)[3], return_dtype=pl.Boolean)
            .alias("is_cspc"),
        ]
    )
    return (processed_data,)


@app.cell
def _(aggregate_to_patient, processed_data):
    patient_level_data = aggregate_to_patient(processed_data)
    return (patient_level_data,)


@app.cell
def _(derive_psa_features, patient_level_data):
    patient_features = derive_psa_features(patient_level_data)
    return (patient_features,)


@app.cell
def _(patient_features, pl):
    # Capture missingness before filling — null grade_group_max means no biopsy
    # was performed (patient is confirmed negative, not just unsampled).
    # Null psad residuals after PSA derivation are already flagged by psa_missing.
    patient_features_clean = patient_features.with_columns(
        [pl.col("grade_group_max").is_null().alias("grade_group_missing")]
    ).with_columns(
        [
            pl.col("psad").fill_null(0.0),
            pl.col("grade_group_max").fill_null(0),
        ]
    )
    return (patient_features_clean,)


@app.cell
def _(
    apply_scaler,
    build_feature_matrix,
    fit_scaler,
    patient_features_clean,
):
    ANALYSIS_FEATURES = [
        "psa",
        "psad",
        "grade_group_max",
        "patient_age",
        "n_mri_studies",
        "had_multiple_mri",
        "psa_missing",
        "grade_group_missing",
    ]
    X, y = build_feature_matrix(patient_features_clean, feature_cols=ANALYSIS_FEATURES)
    scaler = fit_scaler(X, X.columns)
    X_scaled = apply_scaler(X, scaler)
    X_scaled.describe()
    return (ANALYSIS_FEATURES, X_scaled, y)


@app.cell
def _(X_scaled):
    X_scaled.write_csv("data/scaled_feature_matrix.csv")
    return


@app.cell
def _(y):
    y.to_frame().write_csv("data/labels.csv")
    return


@app.cell
def _(X_scaled):
    X_scaled
    return


if __name__ == "__main__":
    app.run()
