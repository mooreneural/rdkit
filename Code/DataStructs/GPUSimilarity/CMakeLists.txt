if(RDK_BUILD_CUDA_SUPPORT)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Pascal through Hopper covers most consumer + datacenter GPUs in use today.
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
  endif()
  set(GPUSIMILARITY_SRCS GPUSimilarityCpu.cpp GPUSimilarityCuda.cu)
else()
  set(GPUSIMILARITY_SRCS GPUSimilarityCpu.cpp GPUSimilarityCudaStub.cpp)
endif()

rdkit_library(GPUSimilarity
              ${GPUSIMILARITY_SRCS}
              LINK_LIBRARIES DataStructs RDGeneral)
target_compile_definitions(GPUSimilarity PRIVATE RDKIT_GPUSIMILARITY_BUILD)

if(RDK_BUILD_CUDA_SUPPORT)
  target_compile_definitions(GPUSimilarity PRIVATE RDK_BUILD_CUDA_SUPPORT)
  set_target_properties(GPUSimilarity PROPERTIES
                        CUDA_SEPARABLE_COMPILATION ON)
endif()

rdkit_headers(GPUSimilarity.h DEST DataStructs/GPUSimilarity)

rdkit_catch_test(testGPUSimilarity catch_tests.cpp
                 LINK_LIBRARIES GPUSimilarity DataStructs)
