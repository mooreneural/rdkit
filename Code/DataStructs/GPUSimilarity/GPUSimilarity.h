//
//  Copyright (C) 2026 Clay Moore and other RDKit contributors
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//
/*! \file GPUSimilarity.h

  \brief Bulk Tanimoto similarity over many fingerprints.

  Computes an M x N matrix of Tanimoto similarities between two sets of
  ExplicitBitVect fingerprints. The implementation will use a CUDA backend
  when the library is built with -DRDK_BUILD_CUDA_SUPPORT=ON and a CUDA
  capable device is available at runtime; otherwise it falls back to a
  multi-threaded CPU kernel using the same packed-bit layout.

  All fingerprints in a call must share the same bit length, and that
  length must be a multiple of 64.
*/

#include <RDGeneral/export.h>
#ifndef RDKIT_GPU_SIMILARITY_H
#define RDKIT_GPU_SIMILARITY_H

#include <cstdint>
#include <vector>
#include <string>

class ExplicitBitVect;

namespace RDKit {
namespace GPUSimilarity {

//! Identifies which backend executed the most recent call (or which one would
//! execute the next call).
enum class Backend : int { CPU = 0, CUDA = 1 };

//! Returns true if the library was compiled with CUDA support and a CUDA
//! device is currently visible.
RDKIT_GPUSIMILARITY_EXPORT bool cudaAvailable();

//! Returns the backend that will be used by the next tanimotoMatrix call,
//! given current build options, device visibility, and the
//! RDK_GPU_SIMILARITY_FORCE_CPU environment variable.
RDKIT_GPUSIMILARITY_EXPORT Backend selectedBackend();

//! Number of words (uint64_t) required to store a fingerprint of \c numBits
//! bits. Throws if \c numBits is not a positive multiple of 64.
RDKIT_GPUSIMILARITY_EXPORT std::size_t wordsForBits(std::size_t numBits);

//! Pack a vector of fingerprints into a contiguous, row-major uint64_t buffer.
/*!
  All fingerprints must have the same bit length, and that length must be a
  positive multiple of 64. Bit i of fingerprint k is stored in
  bit (i % 64) of word (k * words + i / 64), where words = numBits / 64.

  \param fps        the fingerprints to pack; must be non-empty
  \param numBits    output: the bit length shared by all fingerprints
  \return the packed buffer, of length fps.size() * (numBits / 64)
*/
RDKIT_GPUSIMILARITY_EXPORT std::vector<std::uint64_t> packFingerprints(
    const std::vector<const ExplicitBitVect *> &fps, std::size_t &numBits);

//! Compute an M x N Tanimoto similarity matrix from pre-packed buffers.
/*!
  \param probes      packed probe fingerprints, length numProbes * words
  \param numProbes   number of probe fingerprints (M)
  \param targets     packed target fingerprints, length numTargets * words
  \param numTargets  number of target fingerprints (N)
  \param words       number of uint64_t words per fingerprint
  \param out         row-major M x N output buffer, length numProbes * numTargets,
                     allocated by the caller. out[i * numTargets + j] receives
                     the Tanimoto coefficient between probe i and target j.
                     If both fingerprints are all-zero, the result is 0.0.
*/
RDKIT_GPUSIMILARITY_EXPORT void tanimotoMatrix(const std::uint64_t *probes,
                                               std::size_t numProbes,
                                               const std::uint64_t *targets,
                                               std::size_t numTargets,
                                               std::size_t words, double *out);

//! \overload Convenience wrapper that packs the inputs and returns the result.
RDKIT_GPUSIMILARITY_EXPORT void tanimotoMatrix(
    const std::vector<const ExplicitBitVect *> &probes,
    const std::vector<const ExplicitBitVect *> &targets,
    std::vector<double> &out);

//! CPU implementation, exposed for testing and for forcing the CPU path
//! regardless of CUDA availability.
RDKIT_GPUSIMILARITY_EXPORT void tanimotoMatrixCpu(const std::uint64_t *probes,
                                                  std::size_t numProbes,
                                                  const std::uint64_t *targets,
                                                  std::size_t numTargets,
                                                  std::size_t words,
                                                  double *out);

//! CUDA implementation. Throws std::runtime_error when the library was not
//! built with CUDA support or no CUDA device is available.
RDKIT_GPUSIMILARITY_EXPORT void tanimotoMatrixCuda(const std::uint64_t *probes,
                                                   std::size_t numProbes,
                                                   const std::uint64_t *targets,
                                                   std::size_t numTargets,
                                                   std::size_t words,
                                                   double *out);

//! Returns a short description of the active CUDA device, or an empty string
//! when CUDA is unavailable.
RDKIT_GPUSIMILARITY_EXPORT std::string cudaDeviceDescription();

}  // namespace GPUSimilarity
}  // namespace RDKit

#endif
