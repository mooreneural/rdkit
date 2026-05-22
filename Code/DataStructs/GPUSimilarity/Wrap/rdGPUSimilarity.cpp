//
//  Copyright (C) 2026 Clay Moore and other RDKit contributors
//
//   @@ All Rights Reserved @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//
//  Python wrapper for the bulk Tanimoto similarity kernel exposed by the
//  RDKitGPUSimilarity library. Returns the M x N similarity matrix as a 2D
//  NumPy array of doubles.

#define PY_ARRAY_UNIQUE_SYMBOL rdgpusimilarity_array_API

#include <RDBoost/python.h>
#include <RDBoost/Wrap.h>
#include <RDBoost/import_array.h>
#include "numpy/arrayobject.h"

#include <DataStructs/BitVects.h>
#include <DataStructs/ExplicitBitVect.h>
#include <DataStructs/GPUSimilarity/GPUSimilarity.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace python = boost::python;

namespace {

std::vector<const ExplicitBitVect *> extractFps(python::object seq,
                                                const char *which) {
  const auto length = python::len(seq);
  std::vector<const ExplicitBitVect *> out;
  out.reserve(length);
  for (Py_ssize_t i = 0; i < length; ++i) {
    python::extract<const ExplicitBitVect *> ex(seq[i]);
    if (!ex.check()) {
      std::string msg = std::string("expected ExplicitBitVect in `") + which +
                        "` at index " + std::to_string(i);
      throw_value_error(msg.c_str());
    }
    out.push_back(ex());
  }
  return out;
}

PyObject *tanimotoMatrixPy(python::object probes, python::object targets,
                           bool useGpu) {
  const auto probeFps = extractFps(probes, "probes");
  const auto targetFps = extractFps(targets, "targets");

  npy_intp dims[2];
  dims[0] = static_cast<npy_intp>(probeFps.size());
  dims[1] = static_cast<npy_intp>(targetFps.size());
  auto *res = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  if (res == nullptr) {
    throw std::runtime_error("failed to allocate Tanimoto matrix");
  }
  if (probeFps.empty() || targetFps.empty()) {
    return PyArray_Return(res);
  }

  std::size_t probeBits = 0;
  std::size_t targetBits = 0;
  std::vector<std::uint64_t> packedProbes;
  std::vector<std::uint64_t> packedTargets;
  try {
    packedProbes = RDKit::GPUSimilarity::packFingerprints(probeFps, probeBits);
    packedTargets =
        RDKit::GPUSimilarity::packFingerprints(targetFps, targetBits);
  } catch (...) {
    Py_DECREF(res);
    throw;
  }
  if (probeBits != targetBits) {
    Py_DECREF(res);
    throw_value_error(
        "probe and target fingerprints must have the same bit length");
  }
  const std::size_t words = RDKit::GPUSimilarity::wordsForBits(probeBits);
  double *out = static_cast<double *>(PyArray_DATA(res));

  try {
    if (useGpu) {
      RDKit::GPUSimilarity::tanimotoMatrixCuda(
          packedProbes.data(), probeFps.size(), packedTargets.data(),
          targetFps.size(), words, out);
    } else {
      RDKit::GPUSimilarity::tanimotoMatrixCpu(
          packedProbes.data(), probeFps.size(), packedTargets.data(),
          targetFps.size(), words, out);
    }
  } catch (...) {
    Py_DECREF(res);
    throw;
  }
  return PyArray_Return(res);
}

PyObject *tanimotoMatrixAuto(python::object probes, python::object targets) {
  return tanimotoMatrixPy(probes, targets,
                          RDKit::GPUSimilarity::selectedBackend() ==
                              RDKit::GPUSimilarity::Backend::CUDA);
}

PyObject *tanimotoMatrixCpuPy(python::object probes, python::object targets) {
  return tanimotoMatrixPy(probes, targets, /*useGpu=*/false);
}

PyObject *tanimotoMatrixCudaPy(python::object probes, python::object targets) {
  if (!RDKit::GPUSimilarity::cudaAvailable()) {
    throw std::runtime_error(
        "no CUDA device available; either rebuild with "
        "-DRDK_BUILD_CUDA_SUPPORT=ON or use the CPU entry point");
  }
  return tanimotoMatrixPy(probes, targets, /*useGpu=*/true);
}

std::string selectedBackendName() {
  switch (RDKit::GPUSimilarity::selectedBackend()) {
    case RDKit::GPUSimilarity::Backend::CUDA:
      return "cuda";
    case RDKit::GPUSimilarity::Backend::CPU:
    default:
      return "cpu";
  }
}

}  // namespace

BOOST_PYTHON_MODULE(cGPUSimilarity) {
  rdkit_import_array();

  python::scope().attr("__doc__") =
      "Bulk Tanimoto similarity over many ExplicitBitVect fingerprints.\n"
      "\n"
      "Returns an M x N NumPy array of double-precision Tanimoto\n"
      "coefficients, where M is the number of probes and N is the number of\n"
      "targets. All fingerprints in a single call must share the same bit\n"
      "length, and that length must be a positive multiple of 64.\n"
      "\n"
      "When the library was built with -DRDK_BUILD_CUDA_SUPPORT=ON and a\n"
      "CUDA device is visible at runtime, the computation runs on the GPU;\n"
      "otherwise it runs on a multi-threaded CPU kernel. Set the\n"
      "environment variable RDK_GPU_SIMILARITY_FORCE_CPU=1 to force the\n"
      "CPU path even when CUDA is available.";

  python::def("BulkTanimotoMatrix", tanimotoMatrixAuto,
              (python::arg("probes"), python::arg("targets")),
              "Compute an M x N Tanimoto similarity matrix between two\n"
              "iterables of ExplicitBitVect fingerprints, choosing the\n"
              "best available backend (CUDA when present, otherwise CPU).");
  python::def("BulkTanimotoMatrixCpu", tanimotoMatrixCpuPy,
              (python::arg("probes"), python::arg("targets")),
              "Force the multi-threaded CPU implementation.");
  python::def("BulkTanimotoMatrixCuda", tanimotoMatrixCudaPy,
              (python::arg("probes"), python::arg("targets")),
              "Force the CUDA implementation; raises if unavailable.");

  python::def("CudaAvailable", &RDKit::GPUSimilarity::cudaAvailable,
              "True if the library was built with CUDA support and a CUDA\n"
              "device is currently visible.");
  python::def("CudaDeviceDescription",
              &RDKit::GPUSimilarity::cudaDeviceDescription,
              "Short description of the active CUDA device, or an empty\n"
              "string when CUDA is unavailable.");
  python::def("SelectedBackend", &selectedBackendName,
              "Name of the backend that BulkTanimotoMatrix would currently\n"
              "use: 'cuda' or 'cpu'.");
}
