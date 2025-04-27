message("      [Sub-Section] Building PYTHON BINDING ...")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

pybind11_add_module(${PYBIND_NAME} unidock_engine.cpp)

target_link_libraries(${PYBIND_NAME} PRIVATE ${LIB_NAME})
target_include_directories(${PYBIND_NAME}
    PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# install(TARGETS unidock_engine DESTINATION .)

message("      [Sub-Section Done]  PYTHON BINDING !")