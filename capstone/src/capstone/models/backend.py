from __future__ import annotations

import enum
from typing import Protocol, runtime_checkable

import numpy as np

from capstone.models.qiskit_backend import QiskitKernelBackend


class BackendType(enum.Enum):
    """Supported quantum circuit execution backends."""

    QISKIT = "qiskit"
    QISKIT_IBM = "qiskit_ibm"
    CUDA_Q = "cuda_q"


@runtime_checkable
class QuantumKernelBackend(Protocol):
    """Protocol that all quantum kernel backends must satisfy.

    Any class implementing n_qubits, backend_name, and compute_kernel_matrix
    is a valid backend — no inheritance required.
    """

    @property
    def n_qubits(self) -> int: ...

    @property
    def backend_name(self) -> str: ...

    def compute_kernel_matrix(
        self,
        x1: np.ndarray,
        x2: np.ndarray | None = None,
    ) -> np.ndarray: ...


def create_backend(
    backend_type: BackendType,
    n_qubits: int,
    reps: int = 1,
    channel: str | None = None,
) -> QuantumKernelBackend:
    """Instantiate and return the requested quantum kernel backend.

    Args:
        backend_type: Which backend to construct.
        n_qubits: Number of qubits (must match the number of input features).
        reps: Number of ZZFeatureMap repetition layers.
        channel: IBM Quantum channel, passed through to QiskitKernelBackend
            when backend_type is QISKIT_IBM.
    Returns:
        A QuantumKernelBackend instance for the requested backend.
    """

    match backend_type:
        case BackendType.QISKIT:
            return QiskitKernelBackend(n_qubits=n_qubits, reps=reps)
        case BackendType.QISKIT_IBM:
            if channel is None:
                raise ValueError("channel is required for QISKIT_IBM")
            return QiskitKernelBackend(n_qubits=n_qubits, reps=reps, channel=channel)
        case BackendType.CUDA_Q:
            # This is imported here because cuda q isn't available on all
            # platorms, only Linux.  If using MacOS/Windows the backend import
            # will fail.
            from capstone.models.cuda_q_backend import CudaQKernelBackend

            return CudaQKernelBackend(n_qubits=n_qubits, reps=reps)
        case _:
            raise ValueError(f"Unknown backend type: {backend_type}")
