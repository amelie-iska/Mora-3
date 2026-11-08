"""Process-local test defaults, loaded before CUDA handles are initialized.

Exact CUDA continuation tests enable deterministic algorithms. PyTorch/CuBLAS
requires a compatible workspace setting before the first CUDA matrix operation.
Keep any explicit caller setting; do not change model/runtime defaults globally.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
