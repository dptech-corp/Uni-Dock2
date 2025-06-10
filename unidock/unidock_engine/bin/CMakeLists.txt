message("   [Section] Building BIN ...")

add_executable(${BIN_NAME}
        unidock/main.cpp
        unidock/main.h
)
target_compile_definitions(${BIN_NAME} PRIVATE VERSION_NUMBER="${VERSION_NUMBER}")
target_link_libraries(${BIN_NAME} PRIVATE yaml-cpp_0.8.0 ${LIB_NAME})
target_include_directories(${BIN_NAME}
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)

install(
        TARGETS ${BIN_NAME}
        RUNTIME DESTINATION bin
)
message("   [Section Done]  BIN !")
