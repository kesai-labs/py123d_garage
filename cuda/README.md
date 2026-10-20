# py123d_garage_cuda

Opt-in CUDA kernels for the TransFuser training step, compiled against the installed torch. Every kernel mirrors a
torch path that stays the fallback and the test oracle: a wrapper launches its kernel only where `registry.supports()`
says the device runs it, and the training startup check compares both paths on a real batch before the first step and
refuses a requested kernel that falls back to torch on any call.

## Layout

```
src/py123d_garage_cuda/
  __init__.py                 loads _C and exports the op wrappers
  _build.py                   written by setup.py: the architectures, torch, CUDA and cuDNN of this build
  csrc/module.cpp             the empty extension module; importing it registers the ops
  transfuser/
    registry.py               what every kernel family needs: capability, sm_90a, shared memory, dtypes, layout
    <network part>.py         the wrappers of one part: fake implementations, autograd formulas, patches
    csrc/arch.cuh             PY123D_SM90(...) and PY123D_SM90A(...)
    csrc/sm80/ptx.cuh         cp.async, ldmatrix, the bf16 mma.sync shapes and the bf16 pair
    csrc/sm80/planner.cuh     the planner kernels' Philox dropout stream and column sums
    csrc/sm90/tma.cuh         mbarrier, TMA tensor and bulk copies, the tensor-map encoder
    csrc/sm80/                kernels any sm_80 or newer GPU runs
    csrc/sm90/                kernels on TMA, mbarrier and bulk copies; sm_90 and newer
    csrc/sm90a/               kernels on wgmma and setmaxnreg; sm_90 only
```

The Python modules are organized by the network part they patch, because a patch swaps one part and its ops may
differ in requirements. The sources are organized by the oldest architecture a file is built for, because that is the
contract every instruction in the file has to honor. Each `csrc/<directory>/ops.cpp` declares the schemas of the ops
implemented in that directory, so the directory of an op states its minimum requirement and
`tests/cuda_unit/transfuser/test_registry_consistency.py` checks the registry against it. An instruction wrapper or a
host helper two kernels need lives in the header of the architecture that introduced it, not in either kernel; the
sm_90a kernels still carry their own wgmma, mbarrier and TMA wrappers until an sm_90 GPU verifies the move.

## Architectures

Every file compiles for every entry of `TORCH_CUDA_ARCH_LIST`. An instruction newer than the architecture being compiled
goes inside `PY123D_SM90(...)` or `PY123D_SM90A(...)` from `csrc/arch.cuh`, which replaces it with `__trap()` on older
architectures. The host side never reaches those traps: `registry.unsupported_reason()` refuses the op on a device below
its capability, on a device the build does not cover, and on a device with too little shared memory, and the wrapper
then runs the torch path. A file whose every kernel needs sm_90 belongs in `sm90/`; one that dispatches between a
plain path and an sm_90 fast path at run time, like `sm80/depthwise_conv.cu`, belongs in the directory of its minimum.

Compute capabilities, not product names: Hopper is sm_90 and sm_90a, a B200 is sm_100a and an RTX 5090 is sm_120.
Kernels built on sm_90a run on sm_90 only; everything else runs on its capability and newer. Per-GPU tuning, like the
cuDNN engine table of `pin_conv_engines`, is data keyed by device name and library version, never a directory.

## Adding a kernel

1. The kernel and its launch code in one `.cu` under the directory of its minimum capability; its schema in that
   directory's `ops.cpp`; its `TORCH_LIBRARY_IMPL` next to the launch code.
1. A registry entry: the family's ops, capability, `sm90a_only`, the largest dynamic shared memory any of its launches
   asks for, dtypes and layout.
1. The wrapper in the module of its network part: the `supports()` gate with the torch fallback, a `note_fallback()`
   with the reason on every fallback branch, `register_fake`, `register_autograd`, and the patch that swaps it into the
   model, listed in `registry.PATCHES`.
1. An example per op in `test_opcheck_sweep.py`, a case per patch in `test_patch_equivalence.py`, and a numerical test
   against the torch path.

`test_registry_consistency.py` fails on a missing step.

## Building and testing

```bash
micromamba install -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 cuda-driver-dev=12.8 libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev cuda-version=12.8 gcc_linux-64=13 gxx_linux-64=13 ninja
TORCH_CUDA_ARCH_LIST="8.0;9.0a;12.0" CUDA_INC_PATH="$CONDA_PREFIX/targets/x86_64-linux/include" pip install --no-build-isolation ./cuda
pytest tests/cuda_unit
```

The `kernels` job of the ci workflow builds the same way with every warning an error, compares every kernel's register
spills from `ptxas -v` with `cuda/ptxas_spills.json` through `.github/scripts/ptxas_spills.py` (a kernel may not spill
more than recorded; `ptxas_spills.py update <build log>` rewrites the baseline after a deliberate change), runs
clang-tidy with `cuda/.clang-tidy` over `.github/scripts/clang_tidy_compile_commands.py`'s compile database, and runs
`tests/cuda_unit` on the runner's GPU, plain, then without the two model-level files under compute-sanitizer's memcheck,
then the opcheck sweep, one small launch per op, under its racecheck, where a warning-level hazard fails the step.
Tests of kernels the GPU does not run skip with the registry's reason.
