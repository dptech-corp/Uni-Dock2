//
// Created by Congcong Liu on 24-10-9.
//

#include "model/model.h"
#include "myutils/myio.h"
#include "task/DockTask.h"
#include "screening.h"
#include "cuda/common.cuh"
#include <spdlog/spdlog.h>

/**
 * @brief Group ligands into groups, for better managing cuda kernels and optimizing performances
 * @param flex_mol_list Flxible molecules
 * @return Groups of ligands
 */
std::vector<std::vector<int>> group_flex_mols(const UDFlexMolList &flex_mol_list) {
    std::vector<std::vector<int>> groups(NGroup + 1);
    bool flag_overflow;
    for (int i = 0; i < flex_mol_list.size(); i ++) {
        flag_overflow = true;
        for (int igroup = 0; igroup < NGroup; ++igroup){
            if (flex_mol_list[i].vina_types.size() <= NatomThresholds[igroup] &&
                flex_mol_list[i].torsions.size() <= NTorsionThresholds[igroup]) {
                groups[igroup].push_back(i);
                flag_overflow = false;
                break;
            }
        }

        if(flag_overflow){
            groups[NGroup].push_back(i);
        }
    }

    // print groups info    
    for (int igroup  = 0; igroup < NGroup + 1; igroup ++){
        spdlog::info("Group {} size: {}", GroupNames[igroup], groups[igroup].size());
    }
    return groups;
}


int predict_gpu_fix(UDFixMol& udfix_mol){
    // Only consider > 1 MB
    int s1 = sizeof(FixMol) + udfix_mol.coords.size() * sizeof(Real);
    // fix_param_cu
    int s2 = sizeof(FixParamVina) + udfix_mol.natom * sizeof(int);

    return (s1 + s2) / 1048576;
}

int predict_gpu_flex(UDFlexMolList& udflex_mols, int exhaustiveness){
    // Only consider > 1 MB
    int nflex = udflex_mols.size();
    int npose = exhaustiveness * nflex;

    int n_range_all_flex = 0, n_rotated_atoms_all_flex = 0;
    int n_dim_all_flex = 0, n_dim_tri_mat_all_flex = 0;
    int size_inter_all_flex = 0, size_intra_all_flex = 0;
    int n_atom_all_flex = 0;
    int n_dihe_all_flex = 0;

    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];

        size_intra_all_flex += m.intra_pairs.size();
        size_inter_all_flex += m.inter_pairs.size(); // all possible pairs

        n_atom_all_flex += m.natom;
        n_dihe_all_flex += m.dihedrals.size();

        int dim = 3 + 4 + m.dihedrals.size();
        n_dim_all_flex += dim;
        n_dim_tri_mat_all_flex += dim * (dim + 1) / 2;
        for (auto t : m.torsions){
            n_rotated_atoms_all_flex += t.rotated_atoms.size();
            n_range_all_flex += t.range_list.size() / 2;
        }
    }


    //
    int s1 = npose * sizeof(FlexPose) + exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex) * sizeof(Real);
    int s2 = nflex * sizeof(FlexTopo) + (n_atom_all_flex + n_dihe_all_flex * 2 + n_dihe_all_flex * 2 +
        n_dihe_all_flex * 2 + n_rotated_atoms_all_flex) * sizeof(int) +
            n_range_all_flex * 2 * sizeof(Real);
    // flex_param_list_cu
    int s3 = nflex * sizeof(FlexParamVina) +
        (size_intra_all_flex + size_inter_all_flex + n_atom_all_flex) * sizeof(int) +
            (size_intra_all_flex + size_inter_all_flex) / 2 * sizeof(Real);
    // aux_pose_cu
    int s4 = STRIDE_POSE * npose * sizeof(FlexPose) +
        STRIDE_POSE * exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex) * sizeof(Real);
    // aux_grads_cu
    int s5 = STRIDE_G * npose * sizeof(FlexPoseGradient) +
        STRIDE_G * exhaustiveness * n_dihe_all_flex * sizeof(Real);

    // aux_hessians_cu
    int s6 = npose * sizeof(FlexPoseHessian) +
        exhaustiveness * n_dim_tri_mat_all_flex * sizeof(Real);
    // aux_forces_cu
    int s7 = npose * sizeof(FlexForce) +
        exhaustiveness * n_atom_all_flex * 3 * sizeof(Real);
    // Clustering
    int npair = exhaustiveness * (exhaustiveness - 1) / 2; // tri-mat with diagonal
    int s8 = npose * sizeof(Real) +
        nflex * exhaustiveness * sizeof(int) +
            nflex * npair * 2 * sizeof(int) +
                nflex * exhaustiveness * sizeof(int);
    int total = (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) / 1048576;
    return total;
}


/**
 * @brief The main entrance of docking process. Apply cross-device or multi-threads strategies here.
 *          Avoiding further penetration.
 * @param dpfix_mol
 * @param dpflex_mols
 * @param fns_flex
 * @param dp_out
 * @param dock_param
 * @param device_max_memory
 */
void run_screening(UDFixMol & dpfix_mol, UDFlexMolList &dpflex_mols, const std::vector<std::string>& fns_flex,
                   const std::string &dp_out, DockParam& dock_param, int device_max_memory,
                   std::string name_json){

    // print docking prameters
    dock_param.show();

    //------------------ Init CUDA RANDOM
    auto groups = group_flex_mols(dpflex_mols);
    spdlog::debug("Group count: {}", groups.size());

    int batch_id = 0;
    int memory_fix = predict_gpu_fix(dpfix_mol);
    device_max_memory -= memory_fix;

    // For each group, divide flex_mol_list into small batches according to GPU Mem Limit
    for (int igroup=0; igroup < groups.size(); igroup ++){
        spdlog::info("@@@@@@@@@@@@@@ Tackling Group: {} @@@@@@@@@@@@@@", GroupNames[igroup]);
        std::vector<int> & group = groups[igroup];
        Config config = Configs[igroup];
        int num_flex_processed = 0;

        // perform batch by batch
        while (num_flex_processed < group.size()) {
            // For each batch
            ++batch_id;
            int batch_size = 0;
            UDFlexMolList batch_flex_mol_list;
            std::vector<std::string> batch_fns_flex;

            int natom_noH = 0;
            while (num_flex_processed + batch_size < group.size()) {
                int imol = num_flex_processed + batch_size;
                auto& m = dpflex_mols[group[imol]];
                batch_flex_mol_list.emplace_back(m);

                // check GPU memory
                if (predict_gpu_flex(batch_flex_mol_list, dock_param.exhaustiveness) < device_max_memory){
                    batch_fns_flex.emplace_back(fns_flex[group[imol]]);
                    int natom_noH_tmp = m.vina_types.size() - std::count(m.vina_types.begin(), m.vina_types.end(), VN_TYPE_H);
                    if (natom_noH_tmp > natom_noH){
                        natom_noH = natom_noH_tmp;
                    }
                    batch_size++;
                }else{
                    batch_flex_mol_list.pop_back();
                    break;
                }
            }

            spdlog::info("Batch {} size: {}", batch_id, batch_size);
            num_flex_processed += batch_size;
            std::string fp_json = genOutFilePathForMol(name_json + "_" + std::to_string(batch_id) + ".json", dp_out);

            // run on this batch. An individual procedure for global_search_with_local_optimize
            spdlog::info("Perform the Task...");
            auto start = std::chrono::high_resolution_clock::now();

            if (dock_param.opt_steps == -1){ // he
                dock_param.opt_steps = unsigned(25 + natom_noH / 3);
            }
            DockTask task(dpfix_mol, batch_flex_mol_list, dock_param, batch_fns_flex, fp_json);
            task.run();
            spdlog::info("Task is done.");

            std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - start;
            spdlog::info("Batch {} running time: {:.1f} ms", batch_id, duration.count());

        }
    }
}



