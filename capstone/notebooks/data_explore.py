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
def _(apply_scaler, build_feature_matrix, fit_scaler, patient_features):
    X, y = build_feature_matrix(patient_features)
    scaler = fit_scaler(X, X.columns)
    X_scaled = apply_scaler(X, scaler)
    X_scaled.describe()
    return (X_scaled,)


@app.cell
def _(X_scaled):
    X_scaled
    return


if __name__ == "__main__":
    app.run()
