
# Use Catch2 for all tests (instead of gtest)
message("      [Sub-Section] Building component tests ...")
set(sources component_test.cpp
)

add_executable(component_tests ${sources})
target_link_libraries(component_tests ${LIB_NAME} catch2_3.7) # libcatch2_3.7.a doesn't have main function


