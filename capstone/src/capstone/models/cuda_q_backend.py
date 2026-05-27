from __future__ import annotations
from math import pi

import numpy as np
from loguru import logger

import cudaq


class CudaQKernelBackend:
    """Quantum kernel backend using Nvidia's CUDA-Q.

    Encodes classical feature vectors using ZZFeatureMap and computes
    the fidelity kernel K(x1, x2) = |<ψ(x1)|ψ(x2)>|² by measuring the
    all-zeros bitstring probability of the U(x1)·U†(x2) fidelity circuit.

    Uses cudaq.make_kernel() (the imperative builder API) so all gate angles
    are baked in as concrete Python floats before CUDA-Q compilation — this
    avoids the MLIR type-conversion errors that occur with @cudaq.kernel
    closures and return-type annotations in CUDA-Q 0.13.0.

    Attributes:
        _n_qubits: Number of qubits, must equal the number of input features.
        _reps: Number of ZZFeatureMap repetition layers.
        _shots: Number of measurement shots per circuit execution.
    """

    def __init__(self, n_qubits: int, reps: int = 1, shots: int = 1024) -> None:
        """Initialize the CUDA-Q kernel backend.

        Args:
            n_qubits: Number of qubits. Must match the number of features
                passed to compute_kernel_matrix.
            reps: Number of ZZFeatureMap repetition layers. Higher values
                increase expressiveness but also circuit depth.
            shots: Number of measurement shots used to estimate each
                kernel entry. More shots reduce statistical noise.
        """
        self._n_qubits = n_qubits
        self._reps = reps
        self._shots = shots

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        return self._n_qubits

    @property
    def reps(self) -> int:
        """Number of ZZFeatureMap repetition layers."""
        return self._reps

    @property
    def shots(self) -> int:
        """Number of measurement shots per circuit execution."""
        return self._shots

    @property
    def backend_name(self) -> str:
        """Identifier string for this backend."""
        return "cuda_q"

    def _build_fidelity_circuit(self, xi: np.ndarray, xj: np.ndarray) -> object:
        """Build the fidelity circuit U(xi)·U†(xj) with all angles baked in.

        Implements U†(xj) manually (reversed gate order, negated RZ angles)
        to avoid cudaq.adjoint, which causes sample-API errors in 0.13.0.

        Args:
            xi: Feature vector for the ket state ψ(xi).
            xj: Feature vector for the bra state ψ(xj).

        Returns:
            A fully-specified cudaq kernel with measurements on all qubits.
        """
        n = self._n_qubits
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(n)

        # U(xi): forward ZZFeatureMap
        for _ in range(self._reps):
            for i in range(n):
                kernel.h(q[i])
            for i in range(n):
                kernel.rz(2.0 * float(xi[i]), q[i])
            for i in range(n):
                for j in range(i + 1, n):
                    kernel.cx(q[i], q[j])
                    kernel.rz(
                        2.0 * (pi - float(xi[i])) * (pi - float(xi[j])),
                        q[j],
                    )
                    kernel.cx(q[i], q[j])

        # U†(xj): inverse ZZFeatureMap — reps in reverse, gates reversed,
        # RZ angles negated. CX is self-inverse so its sign doesn't change.
        for _ in range(self._reps):
            for i in range(n - 1, -1, -1):
                for j in range(n - 1, i, -1):
                    kernel.cx(q[i], q[j])
                    kernel.rz(
                        -2.0 * (pi - float(xj[i])) * (pi - float(xj[j])),
                        q[j],
                    )
                    kernel.cx(q[i], q[j])
            for i in range(n - 1, -1, -1):
                kernel.rz(-2.0 * float(xj[i]), q[i])
            for i in range(n - 1, -1, -1):
                kernel.h(q[i])

        kernel.mz(q)
        return kernel

    def _run_circuit(self, kernel: object) -> float:
        """Run a fidelity circuit and return the all-zeros bitstring probability.

        Args:
            kernel: A fully-specified cudaq kernel with measurements.

        Returns:
            Estimated probability of measuring all-zeros, equal to
            |<ψ(xj)|ψ(xi)>|² — the kernel value.
        """
        result = cudaq.sample(kernel, shots_count=self._shots)
        zero_key = "0" * self._n_qubits
        return result.count(zero_key) / self._shots

    def compute_kernel_matrix(
        self,
        x1: np.ndarray,
        x2: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute a quantum-kernel matrix between two feature sets.

        Each entry K[i, j] = |<ψ(x1[i])|ψ(x2[j])>|² is estimated by
        sampling the fidelity circuit on the CUDA-Q CPU emulator.

        Args:
            x1: First feature matrix of shape (n_samples_1, n_qubits).
            x2: Optional second feature matrix of shape
                (n_samples_2, n_qubits). If omitted, x2 = x1 and the
                method returns a Gram matrix.

        Returns:
            Kernel matrix of shape (n_samples_1, n_samples_2) with
            entries in [0, 1].
        """
        if x2 is None:
            x2 = x1

        logger.debug(f"Computing kernel matrix: x1={x1.shape} x2={x2.shape}")

        output_matrix = np.zeros((len(x1), len(x2)))

        for i in range(len(x1)):
            for j in range(len(x2)):
                circuit = self._build_fidelity_circuit(x1[i], x2[j])
                output_matrix[i, j] = self._run_circuit(circuit)

        return output_matrix
