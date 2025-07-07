message("   [Section] Building TEST ...")

# ---------- Require: GLIBC 2.31
# Set test data directory
set(TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
# macro
add_definitions(-DTEST_DATA_DIR="${TEST_DATA_DIR}")


include_directories(.)

add_subdirectory(unit)
#add_subdirectory(component)

message("   [Section Done] TEST !")
