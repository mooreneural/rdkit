# GPUSimilarity

Bulk Tanimoto similarity over many fingerprints, with an optional CUDA
backend.

## What it does

`tanimotoMatrix` computes an `M x N` matrix of Tanimoto coefficients between
two sets of `ExplicitBitVect` fingerprints. The result is written into a
caller-allocated row-major `double` buffer of length `M * N`, where entry
`[i, j]` is the Tanimoto coefficient between probe `i` and target `j`.

All fingerprints in a single call must share the same bit length, and that
length must be a positive multiple of 64.

When the library is built with `-DRDK_BUILD_CUDA_SUPPORT=ON` and a CUDA device
is visible at runtime, the matrix is computed on the GPU. Otherwise the call
falls back to a multi-threaded CPU kernel that uses the same packed-bit
layout. The CPU path is always built so the public API works on every
machine, and so the GPU path can be checked against it in tests.

Set `RDK_GPU_SIMILARITY_FORCE_CPU=1` in the environment to force the CPU
path even when CUDA is available.

## Build

```
cmake -DRDK_BUILD_CUDA_SUPPORT=ON ...
```

The CMake option is off by default; the module builds and tests the CPU
kernel without CUDA installed. With the option on, `enable_language(CUDA)`
is invoked and a default set of compute capabilities (Pascal through Hopper)
is selected. Override with `-DCMAKE_CUDA_ARCHITECTURES=...` if needed.

## C++ usage

```cpp
#include <DataStructs/GPUSimilarity/GPUSimilarity.h>
#include <DataStructs/ExplicitBitVect.h>

std::vector<const ExplicitBitVect*> probes  = ...;  // M fingerprints
std::vector<const ExplicitBitVect*> targets = ...;  // N fingerprints

std::vector<double> matrix;  // length M * N on return, row-major
RDKit::GPUSimilarity::tanimotoMatrix(probes, targets, matrix);

// matrix[i * targets.size() + j] is Tanimoto(probes[i], targets[j])
```

For a tight inner loop you can pre-pack once and reuse the buffer:

```cpp
std::size_t bits = 0;
auto packed = RDKit::GPUSimilarity::packFingerprints(targets, bits);
const std::size_t words = RDKit::GPUSimilarity::wordsForBits(bits);
// ... later, possibly many times ...
RDKit::GPUSimilarity::tanimotoMatrix(
    probePacked.data(), probes.size(),
    packed.data(),      targets.size(),
    words, out.data());
```

## Kernel design

* Block dim `(16, 16)` = 256 threads. Each block computes a 16 x 16 tile of
  the output matrix.
* Each block cooperatively loads its 16 probe rows and 16 target rows of
  `words` 64-bit values into shared memory.
* Each thread then accumulates a single Tanimoto coefficient by walking the
  shared-memory rows for its `(probe, target)` pair, using `__popcll` for
  popcount of `a & b`, `a`, and `b`.
* Shared memory per block is `(16 + 16) * words * 8` bytes. For a 2048-bit
  fingerprint this is 8 KiB, well within the 48 KiB per-block budget on
  every supported architecture.

The CPU fallback uses scalar `__builtin_popcountll` / `__popcnt64` over the
same packed layout, parallelised by probe row across hardware threads. A
future change will add an AVX-512 / AVX2 popcount path here.
