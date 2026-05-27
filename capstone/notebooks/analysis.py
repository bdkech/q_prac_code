import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    from loguru import logger

    from capstone.analysis import (
        compute_metrics,
        load_analysis_data,
        predict_qsvm,
        print_report,
        split_dataset,
        train_qsvm,
    )
    from capstone.models.backend import BackendType, create_backend

    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
    )
    return (
        BackendType,
        compute_metrics,
        create_backend,
        load_analysis_data,
        predict_qsvm,
        print_report,
        split_dataset,
        train_qsvm,
    )


@app.cell
def _():
    # Reduce this for fast interactive runs — the full 1,476-patient kernel
    # matrix requires ~1M circuit evaluations on AerSimulator.
    # Set to None to use the full dataset.
    n_subsample = 100
    return (n_subsample,)


@app.cell
def _(load_analysis_data, n_subsample):
    from sklearn.model_selection import train_test_split as _tts

    X_full, y_full, feature_names = load_analysis_data(
        "data/scaled_feature_matrix.csv",
        "data/labels.csv",
    )

    if n_subsample is not None:
        _, X, _, y = _tts(
            X_full,
            y_full,
            test_size=n_subsample,
            stratify=y_full,
            random_state=42,
        )
    else:
        X, y = X_full, y_full

    print(f"Using {len(y)} patients, {int(y.sum())} csPCa positive")
    print(f"Features ({len(feature_names)}): {feature_names}")
    return X, feature_names, y


@app.cell
def _(X, split_dataset, y):
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell
def _(BackendType, create_backend, feature_names):
    backend = create_backend(BackendType.QISKIT, n_qubits=len(feature_names))
    print(f"Backend: {backend.backend_name}, qubits: {backend.n_qubits}")
    return (backend,)


@app.cell
def _(X_train, backend, train_qsvm, y_train):
    svm, K_train = train_qsvm(X_train, y_train, backend)
    diag_mean = float(K_train.diagonal().mean())
    print(f"K_train diagonal mean: {diag_mean:.4f} (expected ≈ 1.0)")
    return (svm,)


@app.cell
def _(
    X_train,
    X_val,
    backend,
    compute_metrics,
    predict_qsvm,
    print_report,
    svm,
    y_val,
):
    y_val_pred, y_val_scores = predict_qsvm(X_val, X_train, svm, backend)
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_scores)
    print_report(val_metrics, "val")
    return


@app.cell
def _(
    X_test,
    X_train,
    backend,
    compute_metrics,
    predict_qsvm,
    print_report,
    svm,
    y_test,
):
    y_test_pred, y_test_scores = predict_qsvm(X_test, X_train, svm, backend)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_scores)
    print_report(test_metrics, "test")
    return


if __name__ == "__main__":
    app.run()
