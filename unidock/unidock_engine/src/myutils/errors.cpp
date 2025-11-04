//
// Created by Congcong Liu on 24-10-15.
//
#include "errors.h"
#include <iostream>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"


void check_cuda(cudaError_t cuda_err, char const* const func, const char* const file, int const line){
   if (cuda_err != cudaSuccess){
        // not zero, namely not cudaSuccess
        printf("CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
               static_cast<unsigned int>(cuda_err), cudaGetErrorName(cuda_err), func);
        throw std::runtime_error("CUDA Runtime Error");
    }
}


/**
 * @brief Initialize the logger
 * @param fp_log Logging file path
 * @param level Level of log, 0 for debug, 1 for info, 2 for warning
 */
void init_logger(const std::string& fp_log, int level){
    try{
        // two sinks
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fp_log);

        // combine two sinks into one logger
        std::vector<spdlog::sink_ptr> sinks = {console_sink, file_sink};
        auto combined_logger = std::make_shared<spdlog::logger>("logger", sinks.begin(), sinks.end());

        // default logger
        spdlog::set_default_logger(combined_logger);

        // debug for development, warning for production
        if (level == 0){
            spdlog::set_level(spdlog::level::debug);
        }
        else if (level == 1){
            spdlog::set_level(spdlog::level::info);
        }
        else{
            spdlog::set_level(spdlog::level::warn);
        }

        // format
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%!:%#] %v");
        // flush on error and above to ensure logs are persisted before throws
        spdlog::flush_on(spdlog::level::err);
        if (auto def_logger = spdlog::default_logger()) {
            def_logger->flush_on(spdlog::level::err);
        }
    }
    catch (const spdlog::spdlog_ex& ex){
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
    }
}
