message("   [Section] lib (3rd Party) ...")

add_library(${THIRD_NAME_CATCH2} OBJECT catch2/catch_amalgamated.cpp)
target_compile_definitions(${THIRD_NAME_CATCH2} PRIVATE CATCH_AMALGAMATED_CUSTOM_MAIN)
target_include_directories(${THIRD_NAME_CATCH2} PUBLIC .)

message("   [Section Done] lib (3rd Party) !")
