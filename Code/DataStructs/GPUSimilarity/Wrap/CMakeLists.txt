remove_definitions(-DRDKIT_GPUSIMILARITY_BUILD)
rdkit_python_extension(cGPUSimilarity
                       rdGPUSimilarity.cpp
                       DEST DataStructs
                       LINK_LIBRARIES GPUSimilarity DataStructs)

add_pytest(pyGPUSimilarity
           ${CMAKE_CURRENT_SOURCE_DIR}/testGPUSimilarity.py)
