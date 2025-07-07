
# Use Catch2 for all tests (instead of gtest)
message("      [Sub-Section] Building unit tests ...")
set(sources unit_test.cpp
        cuda/test_bfgs.cpp
        cuda/test_cu_common.cpp

        format/test_json.cpp

        geometry/test_quaternion.cpp
        geometry/test_rotation.cpp

        myutils/test_math.cpp

        score/test_vina.cpp

        search/test_mc.cpp
        myutils/test_matrix.cpp
)

add_executable(${TESTS_NAME} ${sources})
target_link_libraries(${TESTS_NAME} ${LIB_NAME} ${THIRD_NAME_CATCH2})


