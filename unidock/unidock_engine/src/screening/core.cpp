#include <filesystem>
#include <string>
#include <vector> // ligand paths
#include <chrono>
#include <cuda_runtime.h>

#include <unistd.h>
#include "myutils/errors.h"
#include "model/model.h"
#include "format/json.h"
#include "screening/screening.h"

#include "constants/constants.h"
#include "core.h"


void print_sign() {
    // ANSI Shadow
    std::cout << R"(
    ██╗ccc██╗██████╗c██████╗c
    ██║ccc██║██╔══██╗╚════██╗
    ██║ccc██║██║cc██║c█████╔╝
    ██║ccc██║██║cc██║██╔═══╝c
    ╚██████╔╝██████╔╝███████╗
    c╚═════╝c╚═════╝c╚══════╝
    )" << std::endl;

}

void print_version(){
    std::cout << "UD2 C++ Engine Version: " << VERSION_NUMBER << "\n";
}

namespace {
float resolve_max_memory_mb(int gpu_device_id) {
    float max_memory = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE) / 1024 / 1024 * 0.95f;

    int deviceCount = 0;
    checkCUDA(cudaGetDeviceCount(&deviceCount));
    UD2_REQUIRE(deviceCount > 0, "No CUDA device is found!");

    checkCUDA(cudaSetDevice(gpu_device_id));
    spdlog::info("Set GPU device id to {:d}", gpu_device_id);
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    spdlog::info("Available Memory = {:d} MB   Total Memory = {:d} MB",
                 avail / 1024 / 1024, total / 1024 / 1024);
    int max_gpu_memory = avail / 1024 / 1024 * 0.95; // leave 5%
    if (max_gpu_memory > 0 && max_gpu_memory < max_memory) {
        max_memory = static_cast<float>(max_gpu_memory);
    }
    return max_memory;
}

void apply_search_mode(DockParam& dock_param, const std::string& search_mode) {
    if (search_mode.empty() || search_mode == "free") {
        return;
    }
    if (search_mode == "fast") {
        dock_param.exhaustiveness = 128;
        dock_param.mc_steps = 20;
        dock_param.opt_steps = -1;
    } else if (search_mode == "balance") {
        dock_param.exhaustiveness = 256;
        dock_param.mc_steps = 30;
        dock_param.opt_steps = -1;
    } else if (search_mode == "detail") {
        dock_param.exhaustiveness = 512;
        dock_param.mc_steps = 40;
        dock_param.opt_steps = -1;
    } else {
        UD2_FATALF("Not supported search_mode: {} doesn't belong to (fast, balance, detail, free)", search_mode);
    }
}

void apply_bias(DockParam& dock_param, const std::string& bias, Real bias_k) {
    if (bias == "no") {
        dock_param.bias_type = BT_NO;
    } else if (bias == "pos") {
        dock_param.bias_type = BT_POS;
    } else if (bias == "align") {
        dock_param.bias_type = BT_ALIGN;
    } else {
        UD2_FATALF("Not supported bias: {} doesn't belong to (no, pos, align)", bias);
    }
    dock_param.bias_k = bias_k;
}

CoreContext prepare_context_by_input(CoreContext& ipt) {
    CoreContext ctx;

    ctx.task = ipt.task;

    ctx.output_dir = ipt.output_dir;
    UD2_REQUIRE(!ctx.output_dir.empty(), "No output directory is specified!");
    if (!std::filesystem::exists(ctx.output_dir)) {
        try {
            std::filesystem::create_directories(ctx.output_dir);
        } catch (const std::filesystem::filesystem_error& e) {
            UD2_FATALF("Failed to create output directory {}: {}", ctx.output_dir, e.what());
        }
    }

    ctx.name_json = ipt.name_json;
    ctx.max_memory = resolve_max_memory_mb(ipt.gpu_device_id);
    ctx.fix_mol = ipt.fix_mol;
    ctx.flex_mol_list = ipt.flex_mol_list;
    ctx.fns_flex = ipt.fns_flex;

    //
    // ctx.dock_param.seed = ipt.seed;
    // ctx.dock_param.constraint_docking = ipt.constraint_docking;
    // ctx.dock_param.exhaustiveness = ipt.exhaustiveness;
    // ctx.dock_param.randomize = ipt.randomize;
    // ctx.dock_param.mc_steps = ipt.mc_steps;
    // ctx.dock_param.opt_steps = ipt.opt_steps;
    // if (ctx.dock_param.opt_steps < 0){ //heuristic
    //     ctx.dock_param.opt_steps = -1;
    //     spdlog::info("Use heuristic method to decide opt_steps");
    // }
    // apply_search_mode(ctx.dock_param, ipt.search_mode);
    // ctx.dock_param.refine_steps = ipt.refine_steps;
    // ctx.dock_param.num_pose = ipt.num_pose;
    // ctx.dock_param.energy_range = ipt.energy_range;
    // ctx.dock_param.rmsd_limit = ipt.rmsd_limit;
    //
    // ctx.dock_param.constraint_docking = ipt.constraint_docking;
    // if (ctx.dock_param.constraint_docking) {
    //     ctx.dock_param.randomize = false;
    // }
    //
    // ctx.dock_param.box = ipt.box;
    // apply_bias(ctx.dock_param, ipt.bias, ipt.bias_k);

    return ctx;
}


}



int core_pipeline(CoreContext& ipt) {
    try {

        spdlog::info("==================== UD2 Starts! ======================\n");
        auto start = std::chrono::high_resolution_clock::now();

        // ------------------------------- Prepare Context -------------------------------
        auto ctx = prepare_context_by_input(ipt);

        // ------------------------------- Run Task -------------------------------
        if (ctx.task == "screen") { // allow changing every parameter
            spdlog::info("----------------------- RUN Screening -----------------------");
            run_screening(ctx.fix_mol, ctx.flex_mol_list, ctx.fns_flex, ctx.output_dir, ctx.dock_param, ctx.max_memory, ctx.name_json);
        } else if (ctx.task == "score") {
            spdlog::info("----------------------- RUN Only Scoring -----------------------");
            ctx.dock_param.randomize = false;
            ctx.dock_param.exhaustiveness = 1;
            ctx.dock_param.mc_steps = 0;
            ctx.dock_param.opt_steps = 0;
            ctx.dock_param.refine_steps = 0;
            ctx.dock_param.num_pose = 1;
            ctx.dock_param.energy_range = 999;
            ctx.dock_param.rmsd_limit = 999;
            run_screening(ctx.fix_mol, ctx.flex_mol_list, ctx.fns_flex, ctx.output_dir, ctx.dock_param, ctx.max_memory, ctx.name_json);
        } else if (ctx.task == "benchmark_one") {
            spdlog::warn("benchmark task is not implemented");
            spdlog::info("----------------------- RUN Benchmark on One-Crystal-Ligand Cases -----------------------");
            spdlog::info("----------------------- Given poses are deemed as reference poses -----------------------");
            spdlog::info("----------------------- NOT Loaded Yet -----------------------");
        } else if (ctx.task == "mc") {
            ctx.dock_param.randomize = true;
            ctx.dock_param.opt_steps = 0;
            ctx.dock_param.refine_steps = 0;
            spdlog::info("----------------------- RUN Only Monte Carlo Random Walking (With Clustering) -----------------------");
            run_screening(ctx.fix_mol, ctx.flex_mol_list, ctx.fns_flex, ctx.output_dir, ctx.dock_param, ctx.max_memory, ctx.name_json);
        } else if (ctx.task == "randomize") {
            ctx.dock_param.randomize = true;
            ctx.dock_param.mc_steps = 0;
            ctx.dock_param.opt_steps = 0;
            ctx.dock_param.refine_steps = 0;
            ctx.dock_param.num_pose = ctx.dock_param.exhaustiveness;
            ctx.dock_param.energy_range = 1e9;
            ctx.dock_param.rmsd_limit = 0.;
            spdlog::info("----------------------- RUN Only Randomization (No Clustering) -----------------------");
            run_screening(ctx.fix_mol, ctx.flex_mol_list, ctx.fns_flex, ctx.output_dir, ctx.dock_param, ctx.max_memory, ctx.name_json);
        } else if (ctx.task == "optimize") {
            ctx.dock_param.randomize = false;
            ctx.dock_param.exhaustiveness = 1;
            ctx.dock_param.mc_steps = 0;
            ctx.dock_param.opt_steps = 0;
            ctx.dock_param.num_pose = 1;
            ctx.dock_param.energy_range = 1e9;
            ctx.dock_param.rmsd_limit = 0.;
            spdlog::info("----------------------- RUN Only Optimization on Input Pose (for `refine_steps) -----------------------");
            run_screening(ctx.fix_mol, ctx.flex_mol_list, ctx.fns_flex, ctx.output_dir, ctx.dock_param, ctx.max_memory, ctx.name_json);
        } else {
            UD2_FATALF("Not supported task: {} doesn't belong to (screen, local_only, mc)", ctx.task);
        }

        std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
        spdlog::info("UD2 Total Cost: {:.1f} ms", duration.count());
        return 0;

    } catch (const std::exception& e) {
        spdlog::critical("Core pipeline failed: {}", e.what());
        return 1;

    }
}
