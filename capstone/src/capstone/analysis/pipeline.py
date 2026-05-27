from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from capstone.models.backend import QuantumKernelBackend


def load_analysis_data(
    features_path: str | Path,
    labels_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the scaled feature matrix and labels as numpy arrays.

    Missing values in the feature CSV are structural zeros whose semantic is
    preserved by indicator columns (psa_missing, grade_group_missing). Any
    residual nulls are filled with 0.0 so the kernel backend receives valid
    floats — this is not imputation, the indicator columns carry the meaning.

    Args:
        features_path: Path to scaled_feature_matrix.csv.
        labels_path: Path to labels.csv containing a ``label_cspc`` column.

    Returns:
        Tuple (X, y, feature_names) where X has shape (n_patients, n_features)
        as float64 and y has shape (n_patients,) as int32 binary labels.

    Raises:
        FileNotFoundError: If either CSV path does not exist.
        ValueError: If the two files contain a different number of rows.
    """
    features_path = Path(features_path)
    labels_path = Path(labels_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    X_df = pl.read_csv(features_path)
    y_df = pl.read_csv(labels_path)

    if len(X_df) != len(y_df):
        raise ValueError(
            f"Feature matrix ({len(X_df)} rows) and labels ({len(y_df)} rows) "
            "have different lengths — files may be out of sync"
        )

    feature_names = X_df.columns
    X = X_df.fill_null(0.0).to_numpy(allow_copy=True).astype(np.float64)
    y = y_df["label_cspc"].to_numpy().astype(np.int32)

    logger.debug(
        f"Loaded {X.shape[0]} patients, {X.shape[1]} features, "
        f"{int(y.sum())} csPCa positive ({100 * y.mean():.1f}%)"
    )
    return X, y, list(feature_names)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Stratified 70/15/15 train / val / test split.

    Stratification preserves the csPCa positive rate across all three splits.
    Test fraction is inferred as 1 - train_frac - val_frac.

    Args:
        X: Feature matrix of shape (n, d).
        y: Binary label vector of shape (n,).
        train_frac: Proportion of data for the training set.
        val_frac: Proportion for the validation set.
        random_state: Random seed for reproducibility.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    test_frac = 1.0 - train_frac - val_frac
    # val_of_remainder is the val fraction relative to the (train + val) pool
    val_of_remainder = val_frac / (train_frac + val_frac)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_of_remainder,
        stratify=y_temp,
        random_state=random_state,
    )

    logger.debug(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_qsvm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    backend: QuantumKernelBackend,
) -> tuple[SVC, np.ndarray]:
    """Compute the training quantum kernel matrix and fit a precomputed-kernel SVC.

    K(xi, xj) = |⟨ψ(xi)|ψ(xj)⟩|² replaces the classical kernel. Sklearn's
    SVC with kernel='precomputed' accepts the explicit (n_train, n_train) matrix
    instead of raw features.

    Args:
        X_train: Training features of shape (n_train, n_features).
            n_features must equal backend.n_qubits.
        y_train: Binary training labels of shape (n_train,).
        backend: Quantum kernel backend satisfying QuantumKernelBackend.

    Returns:
        (svm, K_train) — fitted SVC and the (n_train, n_train) kernel matrix.

    Raises:
        ValueError: If X_train column count does not match backend.n_qubits.
    """
    if X_train.shape[1] != backend.n_qubits:
        raise ValueError(
            f"Feature count {X_train.shape[1]} != backend.n_qubits {backend.n_qubits}"
        )

    logger.debug(
        f"Computing training kernel matrix ({len(X_train)}×{len(X_train)}) "
        f"on {backend.backend_name}…"
    )
    K_train = backend.compute_kernel_matrix(X_train)

    svm = SVC(kernel="precomputed", probability=True)
    svm.fit(K_train, y_train)
    logger.debug("SVM trained")
    return svm, K_train


def predict_qsvm(
    X_test: np.ndarray,
    X_train: np.ndarray,
    svm: SVC,
    backend: QuantumKernelBackend,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the test kernel matrix and predict with a trained QSVM.

    K_test has shape (n_test, n_train) — each row is the kernel of one test
    point against all training support vectors, which is the format sklearn's
    precomputed SVC expects at prediction time.

    Args:
        X_test: Test or validation features of shape (n_test, n_features).
        X_train: Training features used to fit the SVM, shape (n_train, n_features).
        svm: Fitted SVC returned by train_qsvm.
        backend: Same backend instance used during training.

    Returns:
        (y_pred, y_scores) — integer class predictions and decision function
        values (used for AUC computation).
    """
    logger.debug(f"Computing test kernel matrix ({len(X_test)}×{len(X_train)})…")
    K_test = backend.compute_kernel_matrix(X_test, X_train)

    y_pred: np.ndarray = svm.predict(K_test)
    y_scores: np.ndarray = svm.decision_function(K_test)
    return y_pred, y_scores
