//
//  Copyright (C) 2026 Clay Moore and other RDKit contributors
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//
//  Compiled when RDK_BUILD_CUDA_SUPPORT is OFF. Provides definitions for
//  the CUDA-facing entry points that simply report unavailability.

#include "GPUSimilarity.h"

#include <stdexcept>
#include <string>

namespace RDKit {
namespace GPUSimilarity {

bool cudaAvailable() { return false; }

std::string cudaDeviceDescription() { return std::string(); }

void tanimotoMatrixCuda(const std::uint64_t * /*probes*/,
                        std::size_t /*numProbes*/,
                        const std::uint64_t * /*targets*/,
                        std::size_t /*numTargets*/, std::size_t /*words*/,
                        double * /*out*/) {
  throw std::runtime_error(
      "RDKit was built without CUDA support; rebuild with "
      "-DRDK_BUILD_CUDA_SUPPORT=ON to use tanimotoMatrixCuda");
}

}  // namespace GPUSimilarity
}  // namespace RDKit
