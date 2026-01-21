message("      [Sub-Section] Building PYTHON BINDING ...")

set(PYBIND_NAME pipeline)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF) # avoid the symbol: fatbinData
pybind11_add_module(${PYBIND_NAME} pipeline.cpp)
install(TARGETS ${PYBIND_NAME} LIBRARY DESTINATION .)

target_link_libraries(${PYBIND_NAME} PRIVATE ${LIB_NAME})
target_include_directories(${PYBIND_NAME}
    PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# ============ Test Runner (for CLion Run button) ============
# A dummy executable that depends on pipeline, then runs Python test script
add_executable(test_pipeline_runner test_runner.cpp)
add_dependencies(test_pipeline_runner ${PYBIND_NAME})
# Pass the .so directory and test script path to the runner
target_compile_definitions(test_pipeline_runner PRIVATE
    PIPELINE_SO_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    TEST_SCRIPT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test_pipeline.py"
    PYTHON_EXECUTABLE="${Python_EXECUTABLE}"
)




# install not specified here; managed by conda

# lib rpath: suppose conda_env/lib/pythonX.Y/site-packages/my_package/unidock_engine.AABB.so
#set_target_properties(${PYBIND_NAME} PROPERTIES
#        INSTALL_RPATH "$ORIGIN/../../../../"
#)

message("      [Sub-Section Done]  PYTHON BINDING !")