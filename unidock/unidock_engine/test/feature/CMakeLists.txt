message("      [Sub-Section] Building Feature Tests ...")

# find all cases
set(CASE_DIR cases)
file(GLOB TEST_CASES
        RELATIVE
        "${CMAKE_CURRENT_SOURCE_DIR}/${CASE_DIR}"
        "${CASE_DIR}/*"
)

# add CTest
foreach(CASE ${TEST_CASES})
    set(TEMP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${CASE_DIR}/${CASE})
    add_test(
            NAME feature_${CASE}
            COMMAND ${TEMP_DIR}/run_test.sh
            $<TARGET_FILE:${BIN_NAME}>  # binary path
            ${TEMP_DIR} # case directory path
            WORKING_DIRECTORY "${TEMP_DIR}"  # perform ud2 inside the case
    )

    # overtime
    set_tests_properties(feature_${CASE} PROPERTIES TIMEOUT 10)
endforeach()


message("      [Sub-Section Done] Feature Tests!")
