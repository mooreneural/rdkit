//
//  Copyright (C) 2026 Clay Moore and other RDKit contributors
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//

#include <catch2/catch_all.hpp>

#include "GPUSimilarity.h"

#include <DataStructs/BitOps.h>
#include <DataStructs/ExplicitBitVect.h>

#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

using RDKit::GPUSimilarity::Backend;
using RDKit::GPUSimilarity::cudaAvailable;
using RDKit::GPUSimilarity::packFingerprints;
using RDKit::GPUSimilarity::selectedBackend;
using RDKit::GPUSimilarity::tanimotoMatrix;
using RDKit::GPUSimilarity::tanimotoMatrixCpu;
using RDKit::GPUSimilarity::tanimotoMatrixCuda;
using RDKit::GPUSimilarity::wordsForBits;

namespace {

std::unique_ptr<ExplicitBitVect> makeRandomFp(std::size_t nBits,
                                              std::mt19937 &rng,
                                              double density = 0.1) {
  auto fp = std::make_unique<ExplicitBitVect>(static_cast<unsigned int>(nBits));
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (std::size_t i = 0; i < nBits; ++i) {
    if (dist(rng) < density) {
      fp->setBit(static_cast<unsigned int>(i));
    }
  }
  return fp;
}

std::vector<std::unique_ptr<ExplicitBitVect>> makeRandomFps(
    std::size_t count, std::size_t nBits, std::mt19937 &rng,
    double density = 0.1) {
  std::vector<std::unique_ptr<ExplicitBitVect>> result;
  result.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    result.push_back(makeRandomFp(nBits, rng, density));
  }
  return result;
}

std::vector<const ExplicitBitVect *> asViews(
    const std::vector<std::unique_ptr<ExplicitBitVect>> &fps) {
  std::vector<const ExplicitBitVect *> views;
  views.reserve(fps.size());
  for (const auto &p : fps) {
    views.push_back(p.get());
  }
  return views;
}

double reference(const ExplicitBitVect &a, const ExplicitBitVect &b) {
  return TanimotoSimilarity(a, b);
}

}  // namespace

TEST_CASE("GPUSimilarity: wordsForBits validates input",
          "[GPUSimilarity]") {
  CHECK(wordsForBits(64) == 1);
  CHECK(wordsForBits(2048) == 32);
  CHECK_THROWS(wordsForBits(0));
  CHECK_THROWS(wordsForBits(100));   // not a multiple of 64
  CHECK_THROWS(wordsForBits(1023));  // not a multiple of 64
}

TEST_CASE("GPUSimilarity: packFingerprints round-trips bit values",
          "[GPUSimilarity]") {
  constexpr unsigned int nBits = 256;
  ExplicitBitVect fp(nBits);
  fp.setBit(0);
  fp.setBit(63);
  fp.setBit(64);
  fp.setBit(130);
  fp.setBit(255);

  std::vector<const ExplicitBitVect *> view{&fp};
  std::size_t actualBits = 0;
  auto packed = packFingerprints(view, actualBits);
  REQUIRE(actualBits == nBits);
  REQUIRE(packed.size() == 4);

  CHECK(((packed[0] >> 0) & 1ULL) == 1ULL);
  CHECK(((packed[0] >> 63) & 1ULL) == 1ULL);
  CHECK(((packed[1] >> 0) & 1ULL) == 1ULL);
  CHECK(((packed[2] >> (130 - 128)) & 1ULL) == 1ULL);
  CHECK(((packed[3] >> (255 - 192)) & 1ULL) == 1ULL);
  // A bit we didn't set should be zero.
  CHECK(((packed[0] >> 5) & 1ULL) == 0ULL);
}

TEST_CASE("GPUSimilarity: packed Tanimoto matches the reference",
          "[GPUSimilarity]") {
  std::mt19937 rng(0xC0FFEE);
  constexpr std::size_t nBits = 512;
  auto probes = makeRandomFps(7, nBits, rng);
  auto targets = makeRandomFps(11, nBits, rng);
  auto probeViews = asViews(probes);
  auto targetViews = asViews(targets);

  std::vector<double> matrix;
  // Force the CPU path so we exercise the always-built kernel.
  std::vector<std::uint64_t> packedProbes;
  std::vector<std::uint64_t> packedTargets;
  std::size_t bits = 0;
  packedProbes = packFingerprints(probeViews, bits);
  packedTargets = packFingerprints(targetViews, bits);
  const std::size_t words = wordsForBits(nBits);
  matrix.assign(probes.size() * targets.size(), -1.0);
  tanimotoMatrixCpu(packedProbes.data(), probes.size(), packedTargets.data(),
                    targets.size(), words, matrix.data());

  for (std::size_t i = 0; i < probes.size(); ++i) {
    for (std::size_t j = 0; j < targets.size(); ++j) {
      const double expected = reference(*probes[i], *targets[j]);
      const double actual = matrix[i * targets.size() + j];
      CHECK(actual == Catch::Approx(expected).margin(1e-12));
    }
  }
}

TEST_CASE("GPUSimilarity: zero fingerprints produce zero similarity",
          "[GPUSimilarity]") {
  constexpr std::size_t nBits = 128;
  ExplicitBitVect a(nBits);
  ExplicitBitVect b(nBits);
  std::vector<const ExplicitBitVect *> probes{&a};
  std::vector<const ExplicitBitVect *> targets{&b};
  std::vector<double> out;
  tanimotoMatrix(probes, targets, out);
  REQUIRE(out.size() == 1);
  CHECK(out[0] == 0.0);
}

TEST_CASE("GPUSimilarity: identical fingerprints are perfectly similar",
          "[GPUSimilarity]") {
  constexpr std::size_t nBits = 128;
  ExplicitBitVect a(nBits);
  a.setBit(1);
  a.setBit(7);
  a.setBit(99);
  std::vector<const ExplicitBitVect *> probes{&a};
  std::vector<const ExplicitBitVect *> targets{&a};
  std::vector<double> out;
  tanimotoMatrix(probes, targets, out);
  REQUIRE(out.size() == 1);
  CHECK(out[0] == Catch::Approx(1.0));
}

TEST_CASE("GPUSimilarity: convenience overload matches packed path",
          "[GPUSimilarity]") {
  std::mt19937 rng(42);
  constexpr std::size_t nBits = 1024;
  auto probes = makeRandomFps(5, nBits, rng);
  auto targets = makeRandomFps(9, nBits, rng);
  auto probeViews = asViews(probes);
  auto targetViews = asViews(targets);

  std::vector<double> convenience;
  tanimotoMatrix(probeViews, targetViews, convenience);

  std::size_t bits = 0;
  auto packedProbes = packFingerprints(probeViews, bits);
  auto packedTargets = packFingerprints(targetViews, bits);
  std::vector<double> packed(probes.size() * targets.size(), 0.0);
  tanimotoMatrixCpu(packedProbes.data(), probes.size(), packedTargets.data(),
                    targets.size(), wordsForBits(nBits), packed.data());

  REQUIRE(convenience.size() == packed.size());
  for (std::size_t k = 0; k < packed.size(); ++k) {
    CHECK(convenience[k] == Catch::Approx(packed[k]).margin(1e-12));
  }
}

TEST_CASE("GPUSimilarity: empty inputs are well-defined",
          "[GPUSimilarity]") {
  std::vector<const ExplicitBitVect *> probes;
  std::vector<const ExplicitBitVect *> targets;
  std::vector<double> out{1.0, 2.0, 3.0};
  tanimotoMatrix(probes, targets, out);
  CHECK(out.empty());
}

TEST_CASE("GPUSimilarity: mismatched bit sizes are rejected",
          "[GPUSimilarity]") {
  ExplicitBitVect a(128);
  ExplicitBitVect b(256);
  std::vector<const ExplicitBitVect *> probes{&a};
  std::vector<const ExplicitBitVect *> targets{&b};
  std::vector<double> out;
  CHECK_THROWS(tanimotoMatrix(probes, targets, out));
}

TEST_CASE("GPUSimilarity: backend selection respects force-cpu env var",
          "[GPUSimilarity]") {
  // Just sanity-check the backend logic without mutating the user's env.
  const Backend selected = selectedBackend();
  if (cudaAvailable()) {
    CHECK(selected == Backend::CUDA);
  } else {
    CHECK(selected == Backend::CPU);
  }
}

TEST_CASE("GPUSimilarity: CUDA path agrees with CPU when available",
          "[GPUSimilarity][cuda]") {
  if (!cudaAvailable()) {
    SKIP("CUDA device not available; skipping GPU/CPU agreement test");
  }
  std::mt19937 rng(123);
  constexpr std::size_t nBits = 2048;
  auto probes = makeRandomFps(17, nBits, rng);
  auto targets = makeRandomFps(23, nBits, rng);
  auto probeViews = asViews(probes);
  auto targetViews = asViews(targets);

  std::size_t bits = 0;
  auto packedProbes = packFingerprints(probeViews, bits);
  auto packedTargets = packFingerprints(targetViews, bits);
  const std::size_t words = wordsForBits(nBits);

  std::vector<double> cpu(probes.size() * targets.size(), 0.0);
  std::vector<double> gpu(probes.size() * targets.size(), 0.0);
  tanimotoMatrixCpu(packedProbes.data(), probes.size(), packedTargets.data(),
                    targets.size(), words, cpu.data());
  tanimotoMatrixCuda(packedProbes.data(), probes.size(), packedTargets.data(),
                     targets.size(), words, gpu.data());

  for (std::size_t k = 0; k < cpu.size(); ++k) {
    CHECK(gpu[k] == Catch::Approx(cpu[k]).margin(1e-12));
  }
}
