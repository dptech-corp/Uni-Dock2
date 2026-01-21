/**
 * Dummy executable to run Python test script from CLion.
 * 
 * This allows you to:
 * 1. Select "test_pipeline_runner" target in CLion
 * 2. Click Run - it will first build pipeline.so (dependency)
 * 3. Then execute the Python test script with correct PYTHONPATH
 * 
 * Build-time defines (from CMakeLists.txt):
 * - PIPELINE_SO_DIR: Directory containing the built pipeline.so
 * - TEST_SCRIPT_PATH: Path to test_pipeline.py
 * - PYTHON_EXECUTABLE: Python interpreter path
 */

#include <cstdlib>
#include <cstdio>
#include <string>

#ifndef PIPELINE_SO_DIR
#define PIPELINE_SO_DIR "."
#endif

#ifndef TEST_SCRIPT_PATH
#define TEST_SCRIPT_PATH "test_pipeline.py"
#endif

#ifndef PYTHON_EXECUTABLE
#define PYTHON_EXECUTABLE "python"
#endif

int main(int argc, char* argv[]) {
    // Set PYTHONPATH to include the .so directory
    std::string pythonpath = "PYTHONPATH=" + std::string(PIPELINE_SO_DIR);
    
    // Check if there's existing PYTHONPATH
    const char* existing = std::getenv("PYTHONPATH");
    if (existing && existing[0] != '\0') {
        pythonpath += ":" + std::string(existing);
    }
    putenv(const_cast<char*>(pythonpath.c_str()));
    
    printf("=== Test Pipeline Runner ===\n");
    printf("PYTHONPATH: %s\n", PIPELINE_SO_DIR);
    printf("Test script: %s\n", TEST_SCRIPT_PATH);
    printf("Python: %s\n", PYTHON_EXECUTABLE);
    printf("============================\n\n");
    
    // Build the command
    std::string cmd = std::string(PYTHON_EXECUTABLE) + " " + std::string(TEST_SCRIPT_PATH);
    
    // Execute Python test script
    return std::system(cmd.c_str());
}


