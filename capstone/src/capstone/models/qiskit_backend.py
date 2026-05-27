from __future__ import annotations

import numpy as np
from loguru import logger

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_aer import AerSimulator


class QiskitKernelBackend:
    """Quantum kernel backend using Qiskit's AerSimulator.

    Encodes classical feature vectors using ZZFeatureMap and computes
    the fidelity kernel K(x1, x2) = |<ψ(x1)|ψ(x2)>|² by measuring
    the all-zeros bitstring probability of the inverse circuit.

    Attributes:
        _n_qubits: Number of qubits, must equal the number of input features.
        _reps: Number of ZZFeatureMap repetition layers.
        _shots: Number of measurement shots per circuit execution.
        _feature_map: Parametrized ZZFeatureMap circuit.
        _simulator: AerSimulator instance for local circuit execution.
    """

    def __init__(
        self,
        n_qubits: int,
        reps: int = 1,
        shots: int = 1024,
        channel: str | None = None,
    ):
        """Initialize the Qiskit kernel backend.

        Args:
            n_qubits: Number of qubits. Must match the number of features
                passed to compute_kernel_matrix.
            reps: Number of ZZFeatureMap repetition layers. Higher values
                increase expressiveness but also circuit depth.
            shots: Number of measurement shots used to estimate each
                kernel entry. More shots reduce statistical noise.
            channel: IBM Quantum channel, e.g. "ibm_quantum". When provided,
                circuits run on the least-busy real IBM backend for that channel.
                When None, falls back to the local AerSimulator.
            ibm_backend: Name of the IBM backend to target, e.g.
              "ibm_brisbane".
        """
        self._n_qubits = n_qubits
        self._reps = reps
        self._shots = shots
        self._feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits, reps=self.reps
        )
        self._channel = channel
        if channel is not None:
            service = QiskitRuntimeService(channel=self._channel)
            self._ibm_backend = service.least_busy(min_num_qubits=self.n_qubits)
        else:
            self._ibm_backend = None
            self._simulator = AerSimulator()

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def reps(self) -> int:
        return self._reps

    @property
    def shots(self) -> int:
        return self._shots

    @property
    def backend_name(self) -> str:
        if self._ibm_backend is not None:
            return self._ibm_backend.name
        else:
            return "qiskit_aer"

    def _build_fidelity_circuit(self, xi: np.ndarray, xj: np.ndarray) -> QuantumCircuit:
        # bind xi to the feature map to get a concrete circuit for data point i
        circuit_i = self._feature_map.assign_parameters(xi)

        # bind xj to the feature map to get a concrete circuit for data point j
        circuit_j = self._feature_map.assign_parameters(xj)

        # take the inverse of circuit j (this is U†(xj))
        circuit_j_inv = circuit_j.inverse()

        # compose: append circuit_j_inv onto circuit_i
        # result is U(xi) · U†(xj)
        fidelity = circuit_i.compose(circuit_j_inv)

        # add measurements to all qubits
        fidelity.measure_all()

        # return the circuit
        return fidelity

    def _run_circuit(self, circuit: QuantumCircuit) -> float:
        """Run a compiled fidelity circuit and return the all-zeros probability.

        Args:
            circuit: A fully assembled QuantumCircuit with measurements.

        Returns:
            Estimated probability of measuring the all-zeros bitstring,
            which equals |<0|U(xi)U†(xj)|0>|² — the kernel value.
        """

        if self._ibm_backend is not None:
            transpiled = transpile(circuit, backend=self._ibm_backend)
            sampler = SamplerV2(mode=self._ibm_backend)
            job = sampler.run([transpiled], shots=self._shots)
            counts = job.result()[0].data.meas.get_counts()
            return counts.get("0" * self._n_qubits, 0) / self._shots
        else:
            # transpile the circuit for self._simulator
            transpiled = transpile(circuit, self._simulator)
            # run the transpiled circuit with self._shots shots
            result = self._simulator.run(transpiled, shots=self._shots).result()
            # extract the result counts from the job
            counts = result.get_counts()

            # build the all-zeros bitstring key (e.g. '0000' for 4 qubits)
            # return counts for all-zeros key divided by total shots
            # use .get(zero_state, 0) so missing keys return 0 instead of raising

            return counts.get("0" * self._n_qubits, 0) / self._shots

    def compute_kernel_matrix(
        self,
        x1: np.ndarray,
        x2: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute a quantum-kernel matrix between two feature sets.

        Each entry is estimated as
        ``K[i, j] = |<psi(x1[i]) | psi(x2[j])>|^2`` by executing the
        corresponding fidelity circuit on the Aer simulator.

        Args:
            x1: First feature matrix of shape ``(n_samples_1, n_qubits)``.
            x2: Optional second feature matrix of shape
                ``(n_samples_2, n_qubits)``. If omitted, ``x2 = x1`` and the
                method returns a Gram matrix.

        Returns:
            Kernel matrix of shape ``(n_samples_1, n_samples_2)`` with
            entries in ``[0, 1]``.
        """
        # if x2 is None, we're computing the Gram matrix — set x2 = x1
        if x2 is None:
            x2 = x1
        logger.debug(f"Computing kernel matrix: x1={x1.shape} x2={x2.shape}")
        # allocate an output matrix of shape (len(x1), len(x2)) filled with zeros
        output_matrix = np.zeros((len(x1), len(x2)))

        # loop over each index i in x1
        for i in range(len(x1)):
            for j in range(len(x2)):
                # build the fidelity circuit for x1[i] and x2[j]
                fidelity_circuit = self._build_fidelity_circuit(x1[i], x2[j])
                # run the circuit and store the result in the output matrix at [i, j]
                output_matrix[i, j] = self._run_circuit(fidelity_circuit)

        return output_matrix
