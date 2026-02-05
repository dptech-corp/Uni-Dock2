message("      [Sub-Section] Building PYTHON BINDING ...")

set(PYBIND_NAME pipeline)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF) # avoid the symbol: fatbinData
pybind11_add_module(${PYBIND_NAME} pipeline.cpp
    ${CMAKE_SOURCE_DIR}/src/format/pybind_parser.cpp)
install(TARGETS ${PYBIND_NAME} LIBRARY DESTINATION .)

target_link_libraries(${PYBIND_NAME} PRIVATE ${LIB_NAME})
target_include_directories(${PYBIND_NAME}
    PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# ============ CTest Integration (pytest) ============
# Run: ctest -R pytest_pipeline -V
# Or in CLion: View -> Tool Windows -> CTest -> right-click pytest_pipeline -> Run
if(BUILD_TEST)
    message("       Building pytest_pipeline test ...")
    add_test(NAME pytest_pipeline
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} -v --tb=short
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(pytest_pipeline PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

message("      [Sub-Section Done]  PYTHON BINDING !")