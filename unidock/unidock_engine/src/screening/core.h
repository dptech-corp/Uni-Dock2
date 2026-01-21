#ifndef UD2_PIPELINE_CORE_H
#define UD2_PIPELINE_CORE_H

#include <string>
#include <vector>
#include <iostream>
#include "model/model.h"


struct CoreInput {
    int seed = 12345;
    bool constraint_docking = false;
    int exhaustiveness = 512;
    bool randomize = true;
    int mc_steps = 20; // MC steps
    int opt_steps = -1; // optimization steps in MC process. Zero if only pure MC search is required.
    int refine_steps = 5; // optimization steps after MC, namely a pure local refinement
    int num_pose = 10;
    Real energy_range = 10.0;
    Real rmsd_limit = 1.0; // a limit to judge whether two poses are the same during clustering
    std::string bias = "no";
    Real bias_k = 0.1;
    std::string task = "screen";
    std::string search_mode = "balance";
    std::string output_dir;
    std::string name_json;
    int gpu_device_id = 0;
    Box box;
    UDFixMol fix_mol;
    UDFlexMolList flex_mol_list;
    std::vector<std::string> fns_flex;
};


struct CoreContext {
    std::string task;
    std::string output_dir;
    std::string name_json;
    float max_memory = 0.0f;
    UDFixMol fix_mol;
    UDFlexMolList flex_mol_list;
    std::vector<std::string> fns_flex;
    DockParam dock_param;

    int gpu_device_id = 0;

    int seed = 12345;
    bool constraint_docking = false;
    int exhaustiveness = 512;
    bool randomize = true;
    int mc_steps = 20; // MC steps
    int opt_steps = -1; // optimization steps in MC process. Zero if only pure MC search is required.
    int refine_steps = 5; // optimization steps after MC, namely a pure local refinement
    int num_pose = 10;
    Real energy_range = 10.0;
    Real rmsd_limit = 1.0; // a limit to judge whether two poses are the same during clustering
    std::string bias = "no";
    Real bias_k = 0.1;
    std::string search_mode = "balance";

};

int core_pipeline(CoreContext& ctx);
// int core_pipeline(CoreInput& ipt);



void print_sign();

void print_version();


#endif // UD2_PIPELINE_CORE_H

