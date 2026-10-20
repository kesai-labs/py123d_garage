// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// Every file compiles for every architecture of the build; an instruction newer than the architecture traps where it
// would run, and the registry never launches that kernel there.

#pragma once

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define PY123D_SM90(...) __trap()
#else
#define PY123D_SM90(...) __VA_ARGS__
#endif

#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define PY123D_SM90A(...) __trap()
#else
#define PY123D_SM90A(...) __VA_ARGS__
#endif
