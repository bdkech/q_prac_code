# linux/amd64 required — NVIDIA CUDA-Q has no ARM wheel; Docker on Apple Silicon
# emulates x86_64 via Rosetta 2 automatically.
FROM --platform=linux/amd64 nvcr.io/nvidia/quantum/cuda-quantum:cu12-0.13.0

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY smoke_test.py ./

# Force the CPU emulator — no NVIDIA GPU is available on a Mac Docker host.
ENV CUDAQ_DEFAULT_SIMULATOR=qpp-cpu

USER root

# Install uv, then install the project.
# After installing deps (which upgrades numpy to 2.x), reinstall qutip so its
# Cython extensions match the new numpy ABI; the CUDA-Q image ships qutip
# compiled against numpy 1.x which is binary-incompatible with numpy 2.x.
RUN pip install uv && \
    uv pip install --system --break-system-packages -e . && \
    uv pip install --system --break-system-packages --upgrade qutip

CMD ["python3", "smoke_test.py"]
