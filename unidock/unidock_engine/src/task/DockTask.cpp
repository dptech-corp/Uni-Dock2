//
// Created by Congcong Liu on 24-9-23.
//

#include <spdlog/spdlog.h>
#include <algorithm>  // 为std::sort
#include <numeric>  // 为std::iota

#include "DockTask.h"
#include "search/mc.h"
#include "cluster/cluster.h"
#include "format/json.h"
#include "optimize/optimize.h"
#include "score/vina.h"
#include "score/score.h"

void DockTask::run(){
    spdlog::debug("All file names of this Task: ");
    for(auto& f : fns_flex){
        spdlog::debug(f);
    }

    try {
        // compute necessary sizes
        prepare_vina();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to prepare Vina: {}", e.what());
    }

    try {
        alloc_gpu();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to allocate GPU: {}", e.what());
    }

    try {
        run_search();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to run search: {}", e.what());
    }
    // todo: free some unnecessary data


    try {
        run_cluster();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to run cluster: {}", e.what());
    }

    // todo: free some unnecessary data


    if (dock_param.refine_steps > 0){
        try {
            run_refine();
        } catch (const std::exception& e) {
            spdlog::critical("Failed to run refine: {}", e.what());
        }
    }

    // Move to CPU
    try{
        run_filter();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to copy results from GPU to CPU: {}", e.what());
    }

    // Run on CPU
    // todo: if not cluster or not refine?
    try {
        run_score();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to run score: {}", e.what());
    }

    // Run on CPU
    try {
        spdlog::info("Dumping the poses to {}...", fp_json);
        dump_poses();
        spdlog::info("Dumping is done.");
    } catch (const std::exception& e) {
        spdlog::critical("Failed to dump poses: {}", e.what());
    }

    // Run on CPU
    try {
        free_gpu();
    } catch (const std::exception& e) {
        spdlog::critical("Failed to free GPU: {}", e.what());
    }
}

void DockTask::prepare_vina(){
    // compute r1 + r2 values for acceleration

    for (auto& flex_mol : udflex_mols){
        // compute r1 + r2 for all intra pairs
        for (int j = 0; j < flex_mol.intra_pairs.size(); j += 2){
            int i1 = flex_mol.intra_pairs[j];
            int i2 = flex_mol.intra_pairs[j + 1];
            flex_mol.r1_plus_r2_intra.push_back(VN_VDW_RADII[flex_mol.vina_types[i1]] + VN_VDW_RADII[flex_mol.vina_types[i2]]);
        }

        // compute r1 + r2 for all inter pairs
        for (int j = 0; j < flex_mol.inter_pairs.size(); j += 2){
            int i1 = flex_mol.inter_pairs[j];
            int i2 = flex_mol.inter_pairs[j + 1];
            flex_mol.r1_plus_r2_inter.push_back(VN_VDW_RADII[flex_mol.vina_types[i1]] + VN_VDW_RADII[udfix_mol.vina_types[i2]]);
        }
    }
}

/**
 * One Step of MC to mutate?
 */
void DockTask::run_search(){
    spdlog::info("Global Search (MC)...");
    // Perform Metropolis-MC and update the flex_pose_list
    mc_cu(flex_pose_list_cu, flex_topo_list_cu,
          *fix_mol_cu, flex_param_list_cu, *fix_param_cu,
          aux_poses_cu, aux_grads_cu, aux_hessians_cu, aux_forces_cu,
          dock_param.mc_steps, dock_param.opt_steps, nflex, dock_param.exhaustiveness,
          dock_param.seed, dock_param.randomize);
    spdlog::info("Global Search (MC) is done.");
}

/**
 * @brief Remove duplicated poses by rmsd limit. Poses with lower energy is preferred.
 */
void DockTask::run_cluster(){
    // clustering according to rmsd_limit, num_pose. [ energy_range(not loaded)]
    spdlog::info("Clustering {} * {} = {} poses ...", nflex, dock_param.exhaustiveness, nflex * dock_param.exhaustiveness);
    cluster_cu(clustered_pose_inds_cu, &npose_clustered, &clustered_pose_inds_list, flex_pose_list_cu, flex_topo_list_cu,
        aux_list_e_cu, aux_list_cluster_cu, aux_rmsd_ij_cu,
        nflex, dock_param.exhaustiveness, dock_param.rmsd_limit);
    spdlog::info("Clustering is done. {} poses are left.", npose_clustered);
}


void DockTask::run_refine(){
    spdlog::info("Run Refinement (BFGS) for {} steps...", dock_param.refine_steps);
    optimize_cu(flex_pose_list_cu, clustered_pose_inds_cu, flex_topo_list_cu, *fix_mol_cu,
                          flex_param_list_cu, *fix_param_cu,
                          aux_poses_cu, aux_grads_cu, aux_hessians_cu,
                          aux_forces_cu,
                          dock_param.refine_steps, npose_clustered, dock_param.exhaustiveness);
    spdlog::info("Refinement (BFGS) is done.");
}

void DockTask::run_filter(){
    spdlog::info("Filtering by num_pose & energy_range...");

    // copy results to cpu
    cp_to_cpu();

    // Object: all clutered poses
    for (int i = 0; i < nflex; i ++){ // for each flex
        auto& clustered_inds = clustered_pose_inds_list[i];

        // prepare indices
        std::vector<int> sorted_indices(clustered_inds.size()); // find the best num_modes poses for each flex
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0); // 填充0,1,2,...
        std::sort(sorted_indices.begin(), sorted_indices.end(),
        [this, &clustered_inds](int a, int b) { return this->flex_pose_list_res[clustered_inds[a]].energy < this->flex_pose_list_res[clustered_inds[b]].energy; });

        int count = 0;
        Real e_min = flex_pose_list_res[clustered_inds[sorted_indices[0]]].energy;
        std::vector<int> filtered_inds;
        for (auto& j: sorted_indices){
            if(flex_pose_list_res[clustered_inds[j]].energy > e_min + dock_param.energy_range){
                break;
            }
            filtered_inds.push_back(clustered_inds[j]);
            count ++;
            if (count >= dock_param.num_pose){
                break;
            }
        }
        filtered_pose_inds_list.push_back(filtered_inds);
    }
    spdlog::info("Filtering is done.");
}



void DockTask::run_score(){
    spdlog::info("Scoring...");
    // only score once
    Vina v;
    std::string show_format = "{:<10}{:<20}";

    for (int i = 0; i < nflex; i ++){
        auto mol = udflex_mols[i];
        int n_tors = mol.torsions.size();
        if(show_score){
            spdlog::info("-------------------------------------------");
            spdlog::info(show_format, "Rank", "Affinity (kcal/mol)");
            spdlog::info("-------------------------------------------");
        }


        // use center[3] to record intra, inter, penalty
        // then use orientation[4] to record Predicted Free Energy of Binding, Total score, inter(contains penalty) score, conf_independent part
        int j_r1 = filtered_pose_inds_list[i][0];
        score(flex_pose_list_res + j_r1, flex_pose_list_real_res + list_i_real[j_r1 * 2], udfix_mol, mol, dock_param.box);
        Real e_intra_rank1 = flex_pose_list_res[j_r1].center[0];

        int pose_num = 0;
        for (auto& j: filtered_pose_inds_list[i]){

            score(flex_pose_list_res + j, flex_pose_list_real_res + list_i_real[j * 2], udfix_mol, mol, dock_param.box);
            flex_pose_list_res[j].rot_vec[1] = flex_pose_list_res[j].center[0] + flex_pose_list_res[j].center[1] +
                flex_pose_list_res[j].center[2]; // Total

            Real e_inter = flex_pose_list_res[j].rot_vec[1] - e_intra_rank1; // Real adopted inter
            // Free Energy of Binding
            flex_pose_list_res[j].rot_vec[0] = v.vina_conf_indep(e_inter, n_tors);  // Affinity
            flex_pose_list_res[j].rot_vec[3] = flex_pose_list_res[j].rot_vec[0] - e_inter;  // Conf-Independent

            pose_num ++;
            if(show_score){
                spdlog::info(show_format, pose_num, flex_pose_list_res[j].rot_vec[0]);
            }
        }
        if(show_score){
            spdlog::info("-------------------------------------------");
        }
    }

    spdlog::info("Scoring is done.");
}



void DockTask::dump_poses(){
    // prepare flex names

    write_poses_to_json(fp_json, fns_flex, filtered_pose_inds_list,
        flex_pose_list_res, flex_pose_list_real_res, list_i_real);

    // output poses all info to json
}

