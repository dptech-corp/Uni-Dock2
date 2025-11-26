cmake_minimum_required(VERSION 3.18)

# ------------------------------ Options for Users
option(BUILD_BIN "build binaries" ON)
option(BUILD_API "build api" ON)
option(BUILD_TEST "build unit tests" ON)
option(BUILD_DOC "build documents" OFF)

# Precision
option(USE_DOUBLE "If TRUE, use double-precision instead of single-precision for the whole project." FALSE)
# fixme: double seems worse than float

# ------------------------------ Options for Developers
set(INPUT_TILE_SIZE 32)
set(INPUT_BLOCK_SIZE 128)
add_compile_definitions(TILE_SIZE=${INPUT_TILE_SIZE})
add_compile_definitions(BLOCK_SIZE=${INPUT_BLOCK_SIZE})
file(STRINGS "${CMAKE_SOURCE_DIR}/../VERSION" VERSION_NUMBER)
#set(VERSION_NUMBER "0.4.3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math --maxrregcount 64") #  --generate-line-info

# For Building Stage
link_directories(${CMAKE_SOURCE_DIR}/lib)
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

set(PRJ_NAME ud2)
set(LIB_NAME ud_lib)
set(THIRD_NAME_CATCH2 catch2)
set(THIRD_NAME_YAML yaml-cpp)
set(BIN_NAME "ud2")
set(TESTS_NAME unit_test)

# install
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) # default path is
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Installation root" FORCE)
endif()

# ------------------------------ Fixed Settings
# use C++ 17 standard for std::filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# https://en.wikipedia.org/wiki/CUDA#GPUs_supported
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES # At least 70!
        60 # 10
        70 # V100
        75 # Turing: RTX 20, T4
        80 # Ampere: A100, A800, L20
        86 # Ampere consumer: RTX 30
        89 # Ada: RTX 40, L4, L40, L40S
#        90 # Hopper / Blackwell: H100, B100/B200
)
endif()
message(STATUS "Supported Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(${PRJ_NAME}
        LANGUAGES CUDA CXX C)

set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)

find_package(CUDAToolkit REQUIRED)

# Default is a release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (USE_DOUBLE)
    add_compile_definitions(Real=double)
    message(STATUS "Precision: double")
else ()
    add_compile_definitions(Real=float)
    message(STATUS "Precision: float")
endif ()


if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif ()

# ------------------------------ Start building
add_subdirectory("lib")

add_subdirectory("src")

# python binding
if (BUILD_API)
    add_subdirectory("api")
endif ()

if (BUILD_BIN)
    add_subdirectory("bin")
endif ()

if (BUILD_TEST) # Compile libs on the fly
    add_subdirectory("include")
    add_subdirectory("test")
endif ()

if (BUILD_DOC)
    add_subdirectory("doc")
endif ()



