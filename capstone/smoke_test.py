import numpy as np
from capstone.models.backend import BackendType, QuantumKernelBackend, create_backend

x = np.array([[0.1, 0.2], [0.3, 0.4]])

# --- Qiskit backend ---
qiskit_backend = create_backend(BackendType.QISKIT, n_qubits=2, reps=1)
assert isinstance(qiskit_backend, QuantumKernelBackend), (
    "Qiskit: Protocol not satisfied"
)

K_qiskit = qiskit_backend.compute_kernel_matrix(x)
print("Qiskit kernel matrix:\n", K_qiskit)
print("Qiskit diagonal (should be ~1.0):", np.diag(K_qiskit))

# --- CUDA-Q backend ---
cuda_backend = create_backend(BackendType.CUDA_Q, n_qubits=2, reps=1)
assert isinstance(cuda_backend, QuantumKernelBackend), "CUDA-Q: Protocol not satisfied"

K_cuda = cuda_backend.compute_kernel_matrix(x)
print("\nCUDA-Q kernel matrix:\n", K_cuda)
print("CUDA-Q diagonal (should be ~1.0):", np.diag(K_cuda))
