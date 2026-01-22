#ifndef UD2_PIPELINE_CORE_H
#define UD2_PIPELINE_CORE_H

#include <string>
#include <vector>
#include "model/model.h"

// ============ Default values for docking parameters ============
namespace CoreInputDefaults {
    // Inherit defaults from DockParam
    inline const DockParam cid_dock_param{};

    constexpr const char* bias = "no";
    inline const Real bias_k = cid_dock_param.bias_k;
    inline const bool constraint_docking = cid_dock_param.constraint_docking;
    inline const Real energy_range = cid_dock_param.energy_range;
    inline const int exhaustiveness = cid_dock_param.exhaustiveness;
    constexpr int gpu_device_id = 0;
    inline const int mc_steps = cid_dock_param.mc_steps;
    constexpr int max_gpu_memory = 0;
    inline const int num_pose = cid_dock_param.num_pose;
    constexpr const char* name_json = "from_python_obj";
    inline const int opt_steps = cid_dock_param.opt_steps;
    constexpr const char* output_dir = "./res_ud2";
    inline const bool randomize = cid_dock_param.randomize;
    inline const int refine_steps = cid_dock_param.refine_steps;
    inline const Real rmsd_limit = cid_dock_param.rmsd_limit;
    constexpr const char* search_mode = "balance";
    inline const int seed = cid_dock_param.seed;
    constexpr const char* task = "screen";

    constexpr bool use_tor_lib = false;
}


// ============ Documentation strings (single source of truth) ============
namespace CoreInputDocs {
    constexpr const char* bias = "Bias type. Options: no, pos (position), align";
    constexpr const char* bias_k = "Bias scaling coefficient";
    constexpr const char* constraint_docking = "Enable constraint docking (disable translation & orientation DOFs)";
    constexpr const char* energy_range = "Energy range for output poses (kcal/mol)";
    constexpr const char* exhaustiveness = "Number of independent MC runs (roughly proportional to time)";
    constexpr const char* gpu_device_id = "GPU device ID to use";
    constexpr const char* max_gpu_memory = "Max GPU memory in MB (0 for all available)";
    constexpr const char* mc_steps = "Monte Carlo random walk steps per run";
    constexpr const char* name_json = "JSON identifier name for output";
    constexpr const char* num_pose = "Maximum number of output poses per ligand";
    constexpr const char* opt_steps = "Optimization steps after each MC step (-1 for auto)";
    constexpr const char* output_dir = "Output directory for docking results";
    constexpr const char* randomize = "Whether to randomize input pose before global search";
    constexpr const char* refine_steps = "Refinement steps after clustering";
    constexpr const char* rmsd_limit = "RMSD threshold for pose clustering (Angstrom)";
    constexpr const char* search_mode = "Search mode. Options: fast, balance, detail, free";
    constexpr const char* seed = "Random seed for reproducibility";
    constexpr const char* task = "Docking task type. Options: screen, score, mc";

    constexpr const char* use_tor_lib = "Use torsion angle library";
}


struct CoreInput {
    std::string bias = CoreInputDefaults::bias;
    Real bias_k = CoreInputDefaults::bias_k;
    bool constraint_docking = CoreInputDefaults::constraint_docking;
    int exhaustiveness = CoreInputDefaults::exhaustiveness;
    int gpu_device_id = CoreInputDefaults::gpu_device_id;
    Real energy_range = CoreInputDefaults::energy_range;
    int max_gpu_memory = CoreInputDefaults::max_gpu_memory;
    int mc_steps = CoreInputDefaults::mc_steps;
    std::string name_json;
    int num_pose = CoreInputDefaults::num_pose;    
    int opt_steps = CoreInputDefaults::opt_steps;
    std::string output_dir = CoreInputDefaults::output_dir;
    int refine_steps = CoreInputDefaults::refine_steps;
    bool randomize = CoreInputDefaults::randomize;
    Real rmsd_limit = CoreInputDefaults::rmsd_limit;
    int seed = CoreInputDefaults::seed;
    std::string search_mode = CoreInputDefaults::search_mode;
    std::string task = CoreInputDefaults::task;

    Box box;
    UDFixMol fix_mol;
    UDFlexMolList flex_mol_list;
    std::vector<std::string> fns_flex;
};


struct CoreContext {
    int max_memory = 0;
    std::string task;
    std::string output_dir;
    std::string name_json;
    UDFixMol fix_mol;
    UDFlexMolList flex_mol_list;
    std::vector<std::string> fns_flex;
    DockParam dock_param;
};

int core_pipeline(CoreInput& ctx);

void print_sign();

void print_version();


#endif // UD2_PIPELINE_CORE_H

