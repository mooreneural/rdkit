//
//  Copyright (C) 2026 Clay Moore and other RDKit contributors
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//

#include "GPUSimilarity.h"

#include <DataStructs/ExplicitBitVect.h>
#include <RDGeneral/Exceptions.h>

#include <RDGeneral/BoostStartInclude.h>
#include <boost/dynamic_bitset.hpp>
#include <RDGeneral/BoostEndInclude.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512F__)
#define RDK_GPUSIM_HAVE_AVX512_VPOPCNTQ 1
#include <immintrin.h>
#else
#define RDK_GPUSIM_HAVE_AVX512_VPOPCNTQ 0
#endif

namespace RDKit {
namespace GPUSimilarity {

namespace {

inline unsigned int popcount64(std::uint64_t x) {
#if defined(_MSC_VER)
  return static_cast<unsigned int>(__popcnt64(x));
#else
  return static_cast<unsigned int>(__builtin_popcountll(x));
#endif
}

//! Popcount over an array of uint64_t words. Uses AVX-512 VPOPCNTQ when the
//! translation unit was compiled with the corresponding -m / /arch flags;
//! otherwise falls back to the scalar 64-bit popcount intrinsic.
inline unsigned int popcountWords(const std::uint64_t *data,
                                  std::size_t words) {
  std::size_t w = 0;
  unsigned int total = 0;
#if RDK_GPUSIM_HAVE_AVX512_VPOPCNTQ
  __m512i acc = _mm512_setzero_si512();
  for (; w + 8 <= words; w += 8) {
    __m512i v = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(data + w));
    acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(v));
  }
  total += static_cast<unsigned int>(_mm512_reduce_add_epi64(acc));
#endif
  for (; w < words; ++w) {
    total += popcount64(data[w]);
  }
  return total;
}

//! Popcount of (a & b) and (a) summed in one pass; we use this to compute the
//! intersection and the per-target popcount together without re-reading the
//! target words. The probe popcount is computed once per probe outside this
//! function and stored separately.
inline void intersectAndTargetPop(const std::uint64_t *probe,
                                  const std::uint64_t *target,
                                  std::size_t words, unsigned int &interOut,
                                  unsigned int &targetOut) {
  std::size_t w = 0;
  unsigned int inter = 0;
  unsigned int targ = 0;
#if RDK_GPUSIM_HAVE_AVX512_VPOPCNTQ
  __m512i interAcc = _mm512_setzero_si512();
  __m512i targAcc = _mm512_setzero_si512();
  for (; w + 8 <= words; w += 8) {
    __m512i a = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(probe + w));
    __m512i b = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(target + w));
    targAcc = _mm512_add_epi64(targAcc, _mm512_popcnt_epi64(b));
    interAcc = _mm512_add_epi64(interAcc,
                                _mm512_popcnt_epi64(_mm512_and_si512(a, b)));
  }
  inter += static_cast<unsigned int>(_mm512_reduce_add_epi64(interAcc));
  targ += static_cast<unsigned int>(_mm512_reduce_add_epi64(targAcc));
#endif
  for (; w < words; ++w) {
    const std::uint64_t a = probe[w];
    const std::uint64_t b = target[w];
    targ += popcount64(b);
    inter += popcount64(a & b);
  }
  interOut = inter;
  targetOut = targ;
}

bool envFlagSet(const char *name) {
  const char *v = std::getenv(name);
  if (v == nullptr) {
    return false;
  }
  return v[0] != '\0' && v[0] != '0';
}

}  // namespace

std::size_t wordsForBits(std::size_t numBits) {
  if (numBits == 0 || (numBits % 64) != 0) {
    throw ValueErrorException(
        "GPUSimilarity requires fingerprint bit length to be a positive "
        "multiple of 64");
  }
  return numBits / 64;
}

std::vector<std::uint64_t> packFingerprints(
    const std::vector<const ExplicitBitVect *> &fps, std::size_t &numBits) {
  if (fps.empty()) {
    numBits = 0;
    return {};
  }
  if (fps.front() == nullptr || fps.front()->dp_bits == nullptr) {
    throw ValueErrorException("null fingerprint in GPUSimilarity input");
  }
  numBits = fps.front()->getNumBits();
  const std::size_t words = wordsForBits(numBits);

  std::vector<std::uint64_t> packed(fps.size() * words, 0);
  using block_t = boost::dynamic_bitset<>::block_type;
  constexpr std::size_t bitsPerBlock = sizeof(block_t) * 8;
  static_assert(bitsPerBlock == 32 || bitsPerBlock == 64,
                "Unexpected dynamic_bitset block size");

  std::vector<block_t> scratch;
  for (std::size_t k = 0; k < fps.size(); ++k) {
    const ExplicitBitVect *fp = fps[k];
    if (fp == nullptr || fp->dp_bits == nullptr) {
      throw ValueErrorException("null fingerprint in GPUSimilarity input");
    }
    if (fp->getNumBits() != numBits) {
      throw ValueErrorException(
          "all fingerprints must share the same bit length");
    }
    const std::size_t numBlocks = fp->dp_bits->num_blocks();
    if (scratch.size() < numBlocks) {
      scratch.resize(numBlocks);
    }
    boost::to_block_range(*fp->dp_bits, scratch.data());

    std::uint64_t *dest = packed.data() + k * words;
    if constexpr (bitsPerBlock == 64) {
      std::memcpy(dest, scratch.data(), words * sizeof(std::uint64_t));
    } else {
      // 32-bit blocks: combine two adjacent blocks into one uint64_t.
      const std::size_t pairs = words;
      for (std::size_t w = 0; w < pairs; ++w) {
        std::uint64_t lo = (2 * w < numBlocks)
                               ? static_cast<std::uint64_t>(scratch[2 * w])
                               : 0;
        std::uint64_t hi = (2 * w + 1 < numBlocks)
                               ? static_cast<std::uint64_t>(scratch[2 * w + 1])
                               : 0;
        dest[w] = lo | (hi << 32);
      }
    }
  }
  return packed;
}

void tanimotoMatrixCpu(const std::uint64_t *probes, std::size_t numProbes,
                       const std::uint64_t *targets, std::size_t numTargets,
                       std::size_t words, double *out) {
  if (numProbes == 0 || numTargets == 0) {
    return;
  }
  if (probes == nullptr || targets == nullptr || out == nullptr) {
    throw ValueErrorException("null buffer passed to tanimotoMatrixCpu");
  }
  if (words == 0) {
    throw ValueErrorException("words must be > 0");
  }

  unsigned int hwThreads = std::thread::hardware_concurrency();
  if (hwThreads == 0) {
    hwThreads = 1;
  }
  // Cap the worker count so we don't oversubscribe on tiny workloads.
  const std::size_t maxUseful =
      (numProbes + 0) > 0 ? std::min<std::size_t>(numProbes, hwThreads) : 1;
  const std::size_t numWorkers = std::max<std::size_t>(1, maxUseful);

  auto worker = [&](std::size_t probeStart, std::size_t probeEnd) {
    for (std::size_t i = probeStart; i < probeEnd; ++i) {
      const std::uint64_t *pRow = probes + i * words;
      const unsigned int probePop = popcountWords(pRow, words);
      double *outRow = out + i * numTargets;
      for (std::size_t j = 0; j < numTargets; ++j) {
        const std::uint64_t *tRow = targets + j * words;
        unsigned int interPop = 0;
        unsigned int targetPop = 0;
        intersectAndTargetPop(pRow, tRow, words, interPop, targetPop);
        const unsigned int unionPop = probePop + targetPop - interPop;
        outRow[j] =
            unionPop == 0 ? 0.0 : static_cast<double>(interPop) / unionPop;
      }
    }
  };

  if (numWorkers == 1) {
    worker(0, numProbes);
    return;
  }

  std::vector<std::thread> threads;
  threads.reserve(numWorkers - 1);
  const std::size_t chunk = (numProbes + numWorkers - 1) / numWorkers;
  for (std::size_t t = 0; t + 1 < numWorkers; ++t) {
    const std::size_t start = t * chunk;
    const std::size_t end = std::min(start + chunk, numProbes);
    if (start >= end) {
      break;
    }
    threads.emplace_back(worker, start, end);
  }
  // Run the last slice on the calling thread to avoid an extra spawn.
  const std::size_t lastStart = (numWorkers - 1) * chunk;
  if (lastStart < numProbes) {
    worker(lastStart, numProbes);
  }
  for (auto &th : threads) {
    th.join();
  }
}

void tanimotoMatrix(const std::uint64_t *probes, std::size_t numProbes,
                    const std::uint64_t *targets, std::size_t numTargets,
                    std::size_t words, double *out) {
  if (selectedBackend() == Backend::CUDA) {
    tanimotoMatrixCuda(probes, numProbes, targets, numTargets, words, out);
  } else {
    tanimotoMatrixCpu(probes, numProbes, targets, numTargets, words, out);
  }
}

void tanimotoMatrix(const std::vector<const ExplicitBitVect *> &probes,
                    const std::vector<const ExplicitBitVect *> &targets,
                    std::vector<double> &out) {
  out.assign(probes.size() * targets.size(), 0.0);
  if (probes.empty() || targets.empty()) {
    return;
  }
  std::size_t probeBits = 0;
  std::size_t targetBits = 0;
  std::vector<std::uint64_t> packedProbes = packFingerprints(probes, probeBits);
  std::vector<std::uint64_t> packedTargets =
      packFingerprints(targets, targetBits);
  if (probeBits != targetBits) {
    throw ValueErrorException(
        "probe and target fingerprints must have the same bit length");
  }
  const std::size_t words = wordsForBits(probeBits);
  tanimotoMatrix(packedProbes.data(), probes.size(), packedTargets.data(),
                 targets.size(), words, out.data());
}

Backend selectedBackend() {
  if (envFlagSet("RDK_GPU_SIMILARITY_FORCE_CPU")) {
    return Backend::CPU;
  }
  return cudaAvailable() ? Backend::CUDA : Backend::CPU;
}

}  // namespace GPUSimilarity
}  // namespace RDKit
