# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Written by setup.py at build time: the architectures and libraries this build of py123d_garage_cuda targets."""

ARCHITECTURES: tuple[str, ...]
TORCH: str
CUDA: str
CUDNN: int | None
