message("      [Sub-Section] Building Feature Tests ...")

# Feature tests run the ud2 binary, so only register when BUILD_BIN is ON
# Each subdirectory of feature/ is one test case (run full docking, then validate output)
if(BUILD_BIN)
file(GLOB TEST_CASE_ENTRIES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*")
foreach(CASE ${TEST_CASE_ENTRIES})
    set(CASE_DIR_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${CASE}")
    if(IS_DIRECTORY "${CASE_DIR_FULL}" AND EXISTS "${CASE_DIR_FULL}/run_test.sh")
        add_test(
                NAME feature_${CASE}
                COMMAND bash ${CASE_DIR_FULL}/run_test.sh
                $<TARGET_FILE:${BIN_NAME}>  # binary path
                ${CASE_DIR_FULL}            # case directory path
                WORKING_DIRECTORY "${CASE_DIR_FULL}"
        )
        set_tests_properties(feature_${CASE} PROPERTIES TIMEOUT 120)
    endif()
endforeach()
endif()


message("      [Sub-Section Done] Feature Tests!")
