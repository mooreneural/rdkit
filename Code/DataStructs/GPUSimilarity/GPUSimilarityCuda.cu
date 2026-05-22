//
//  Copyright (C) 2026 Clay Moore and other RDKit contributors
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//
//  CUDA implementation of the bulk Tanimoto similarity kernel.
//
//  Layout: a thread block of (TX, TY) = (16, 16) cooperatively loads up to
//  TX target fingerprints and TY probe fingerprints (each W uint64_t words)
//  into shared memory, then each thread accumulates a single Tanimoto
//  coefficient over W popcount-AND / popcount-OR pairs.

#include "GPUSimilarity.h"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace RDKit {
namespace GPUSimilarity {

namespace {

constexpr int kTileX = 16;  // target tile width
constexpr int kTileY = 16;  // probe tile height

void cudaCheck(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::ostringstream os;
    os << "CUDA " << what << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(os.str());
  }
}

bool detectDevice(int &deviceId) {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count <= 0) {
    cudaGetLastError();  // clear sticky error
    return false;
  }
  status = cudaGetDevice(&deviceId);
  if (status != cudaSuccess) {
    cudaGetLastError();
    deviceId = 0;
  }
  return true;
}

__global__ void tanimotoKernel(const std::uint64_t *__restrict__ probes,
                               std::size_t numProbes,
                               const std::uint64_t *__restrict__ targets,
                               std::size_t numTargets, std::size_t words,
                               double *__restrict__ out) {
  extern __shared__ std::uint64_t shmem[];
  std::uint64_t *shProbes = shmem;                       // kTileY * words
  std::uint64_t *shTargets = shmem + kTileY * words;     // kTileX * words

  const int tx = threadIdx.x;  // target slot within tile [0, kTileX)
  const int ty = threadIdx.y;  // probe slot within tile [0, kTileY)
  const int threadsPerBlock = kTileX * kTileY;
  const int linearTid = ty * kTileX + tx;

  const std::size_t probeBase =
      static_cast<std::size_t>(blockIdx.y) * kTileY;
  const std::size_t targetBase =
      static_cast<std::size_t>(blockIdx.x) * kTileX;

  // Cooperative load of probe tile.
  const std::size_t probeWords = kTileY * words;
  for (std::size_t idx = linearTid; idx < probeWords;
       idx += threadsPerBlock) {
    const std::size_t row = idx / words;
    const std::size_t w = idx % words;
    const std::size_t globalRow = probeBase + row;
    shProbes[idx] =
        (globalRow < numProbes) ? probes[globalRow * words + w] : 0ULL;
  }

  // Cooperative load of target tile.
  const std::size_t targetWords = kTileX * words;
  for (std::size_t idx = linearTid; idx < targetWords;
       idx += threadsPerBlock) {
    const std::size_t row = idx / words;
    const std::size_t w = idx % words;
    const std::size_t globalRow = targetBase + row;
    shTargets[idx] =
        (globalRow < numTargets) ? targets[globalRow * words + w] : 0ULL;
  }

  __syncthreads();

  const std::size_t probeIdx = probeBase + static_cast<std::size_t>(ty);
  const std::size_t targetIdx = targetBase + static_cast<std::size_t>(tx);
  if (probeIdx >= numProbes || targetIdx >= numTargets) {
    return;
  }

  unsigned int interPop = 0;
  unsigned int probePop = 0;
  unsigned int targetPop = 0;
  const std::uint64_t *pRow = shProbes + ty * words;
  const std::uint64_t *tRow = shTargets + tx * words;
  for (std::size_t w = 0; w < words; ++w) {
    const std::uint64_t a = pRow[w];
    const std::uint64_t b = tRow[w];
    probePop += __popcll(a);
    targetPop += __popcll(b);
    interPop += __popcll(a & b);
  }
  const unsigned int unionPop = probePop + targetPop - interPop;
  const double result =
      unionPop == 0 ? 0.0 : static_cast<double>(interPop) / unionPop;
  out[probeIdx * numTargets + targetIdx] = result;
}

}  // namespace

bool cudaAvailable() {
  int deviceId = 0;
  return detectDevice(deviceId);
}

std::string cudaDeviceDescription() {
  int deviceId = 0;
  if (!detectDevice(deviceId)) {
    return std::string();
  }
  cudaDeviceProp prop{};
  cudaError_t status = cudaGetDeviceProperties(&prop, deviceId);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return std::string();
  }
  std::ostringstream os;
  os << prop.name << " (cc " << prop.major << "." << prop.minor << ", "
     << (prop.totalGlobalMem >> 20) << " MiB)";
  return os.str();
}

void tanimotoMatrixCuda(const std::uint64_t *probes, std::size_t numProbes,
                        const std::uint64_t *targets, std::size_t numTargets,
                        std::size_t words, double *out) {
  if (numProbes == 0 || numTargets == 0) {
    return;
  }
  if (probes == nullptr || targets == nullptr || out == nullptr) {
    throw std::runtime_error("null buffer passed to tanimotoMatrixCuda");
  }
  if (words == 0) {
    throw std::runtime_error("words must be > 0");
  }
  int deviceId = 0;
  if (!detectDevice(deviceId)) {
    throw std::runtime_error(
        "no CUDA device available for tanimotoMatrixCuda");
  }

  const std::size_t probeBytes = numProbes * words * sizeof(std::uint64_t);
  const std::size_t targetBytes = numTargets * words * sizeof(std::uint64_t);
  const std::size_t outBytes = numProbes * numTargets * sizeof(double);

  std::uint64_t *dProbes = nullptr;
  std::uint64_t *dTargets = nullptr;
  double *dOut = nullptr;
  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&dProbes), probeBytes),
            "malloc(probes)");
  try {
    cudaCheck(cudaMalloc(reinterpret_cast<void **>(&dTargets), targetBytes),
              "malloc(targets)");
    try {
      cudaCheck(cudaMalloc(reinterpret_cast<void **>(&dOut), outBytes),
                "malloc(out)");
      cudaCheck(cudaMemcpy(dProbes, probes, probeBytes, cudaMemcpyHostToDevice),
                "memcpy(probes)");
      cudaCheck(
          cudaMemcpy(dTargets, targets, targetBytes, cudaMemcpyHostToDevice),
          "memcpy(targets)");

      const dim3 block(kTileX, kTileY, 1);
      const dim3 grid(static_cast<unsigned int>(
                          (numTargets + kTileX - 1) / kTileX),
                      static_cast<unsigned int>(
                          (numProbes + kTileY - 1) / kTileY),
                      1);
      const std::size_t shmemBytes =
          (kTileX + kTileY) * words * sizeof(std::uint64_t);

      tanimotoKernel<<<grid, block, shmemBytes>>>(
          dProbes, numProbes, dTargets, numTargets, words, dOut);
      cudaCheck(cudaGetLastError(), "launch(tanimotoKernel)");
      cudaCheck(cudaDeviceSynchronize(), "synchronize");
      cudaCheck(cudaMemcpy(out, dOut, outBytes, cudaMemcpyDeviceToHost),
                "memcpy(out)");

      cudaFree(dOut);
    } catch (...) {
      cudaFree(dTargets);
      cudaFree(dProbes);
      throw;
    }
    cudaFree(dTargets);
  } catch (...) {
    cudaFree(dProbes);
    throw;
  }
  cudaFree(dProbes);
}

}  // namespace GPUSimilarity
}  // namespace RDKit
