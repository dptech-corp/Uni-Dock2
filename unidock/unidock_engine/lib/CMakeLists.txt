message("   [Section] lib (3rd Party) ...")
add_definitions(-DCATCH_AMALGAMATED_CUSTOM_MAIN)

add_library(${THIRD_NAME_CATCH2} OBJECT catch2/catch_amalgamated.cpp)
#target_include_directories(${THIRD_NAME_CATCH2} PUBLIC ${CMAKE_SOURCE_DIR}/include/yaml-cpp yaml-cpp/)

file(GLOB_RECURSE YAML_SOURCES yaml-cpp/*)
add_library(${THIRD_NAME_YAML} OBJECT ${YAML_SOURCES})
target_include_directories(${THIRD_NAME_CATCH2} PUBLIC .)

message("   [Section Done] lib (3rd Party) !")


