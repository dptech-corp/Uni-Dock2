//
// Created by Congcong Liu on 24-9-19.
//
#ifndef ERRORS_H
#define ERRORS_H

#include <string>
#include <cuda_runtime.h>
#include "spdlog/spdlog.h"
#include <cstdlib>
#include <stdexcept>

void check_cuda(cudaError_t cuda_err, char const *const func, const char *const file, int const line);
#define checkCUDA(val) check_cuda((val), #val, __FILE__, __LINE__)

void init_logger(const std::string& fp_log = "ud.log", int level=1);

#define CUDA_ERROR(...) printf("[CUDA error][%s:%d][%s][Block:%d,Thread:%d] ", \
__FILE__, __LINE__, __func__, \
blockIdx.x, threadIdx.x); \
printf(__VA_ARGS__); \
printf("\n")

// ===================== Fatal logging helpers =====================
// UD2_FATALF: log critical message with location and exit
#define UD2_FATALF(fmt_str, ...)                                                       \
    do {                                                                               \
        auto _msg = fmt::format("[{}:{}:{}] " fmt_str, __FILE__, __func__, __LINE__,  \
                                 ##__VA_ARGS__);                                       \
        spdlog::critical("{}", _msg);                                                 \
        throw std::runtime_error(_msg);                                                \
    } while (0)

// UD2_REQUIRE: check condition; on failure, log and exit
#define UD2_REQUIRE(cond, fmt_str, ...)                                                \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            UD2_FATALF(fmt_str, ##__VA_ARGS__);                                       \
        }                                                                              \
    } while (0)

#endif //ERRORS_H