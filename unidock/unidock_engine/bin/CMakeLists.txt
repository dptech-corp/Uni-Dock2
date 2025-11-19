message("   [Section] Building BIN ...")

add_executable(${BIN_NAME}
        unidock/main.cpp
        unidock/main.h
)
target_compile_definitions(${BIN_NAME} PRIVATE VERSION_NUMBER="${VERSION_NUMBER}")
target_link_libraries(${BIN_NAME} PRIVATE ${THIRD_NAME_YAML} ${LIB_NAME})
target_include_directories(${BIN_NAME} PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_SOURCE_DIR}/src
)

install(TARGETS ${BIN_NAME}
        RUNTIME DESTINATION bin
)
message("   [Section Done]  BIN !")

#add_subdirectory(myapp)