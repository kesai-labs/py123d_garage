# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from py123d_garage.common.runtime_typing import import_unwrapped

# Triton compiles the kernel from the function object it receives and resolves names through that function's
# globals and closure; a jaxtyped wrapper hides both, so tl is undefined at compile time.
import_unwrapped("py123d_garage.training.clip_adamw_kernel")
