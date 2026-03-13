//
// Created by Congcong Liu on 24-11-21.
//

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cooperative_groups/reduce.h>
#include "cluster/cluster.h"
#include "myutils/errors.h"
#include "common.cuh"


namespace cg = cooperative_groups;

__device__ __forceinline__ Real cal_rmsd_pair_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                   const FlexPose* pose1, const FlexPose* pose2,
                                                   const int* vn_types, int natom){
    Real rmsd = 0;
    Real tmp = 0;
    int n_heavy = 0;
    for (int i = tile.thread_rank(); i < natom; i += tile.num_threads()){
        if (vn_types[i] == VN_TYPE_H) continue;
        tmp = pose1->coords[i * 3] - pose2->coords[i * 3];
        rmsd += tmp * tmp;
        tmp = pose1->coords[i * 3 + 1] - pose2->coords[i * 3 + 1];
        rmsd += tmp * tmp;
        tmp = pose1->coords[i * 3 + 2] - pose2->coords[i * 3 + 2];
        rmsd += tmp * tmp;
        n_heavy++;
    }
    tile.sync();
    rmsd = cg::reduce(tile, rmsd, cg::plus<Real>());
    n_heavy = cg::reduce(tile, n_heavy, cg::plus<int>());

    return sqrt(rmsd / n_heavy);
}


__global__ void cal_rmsd_matrix(Real* out_rmsd_matrix, const FlexPose* poses,
                                const FlexTopo* list_flex_topo, int E){
    int i = blockIdx.x;
    int j = blockIdx.y;
    int id_flex = blockIdx.z;
    if (i >= j) return;

    auto cta = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(cta);

    const FlexPose& pose1 = poses[id_flex * E + i];
    const FlexPose& pose2 = poses[id_flex * E + j];

    const FlexTopo* topo = list_flex_topo + id_flex;
    Real rmsd = cal_rmsd_pair_warp(tile, &pose1, &pose2, topo->vn_types, topo->natom);
    if (tile.thread_rank() == 0){
        out_rmsd_matrix[id_flex * E * E + i * E + j] = rmsd;
    }
    tile.sync();
}


__global__ void get_pose_energy(const FlexPose* poses, Real* out_list_e, int nflex, int npose){
    int i_thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_thread < nflex * npose){
        out_list_e[i_thread] = poses[i_thread].energy;
    }
}

/**
 * @brief Greedy Leader clustering: GPU computes E*E RMSD matrix, CPU selects leaders.
 *        Poses are sorted by energy; each pose is kept only if it differs from all
 *        already-selected leaders by >= rmsd_limit. This avoids transitive elimination.
 */
void cluster_cu(int* out_clustered_pose_inds_cu, int* out_npose_clustered, std::vector<std::vector<int>>* clustered_pose_inds_list,
                const FlexPose* poses_cu, const FlexTopo* list_flex_topo,
                Real* aux_list_e_cu, Real* aux_rmsd_matrix_cu,
                int nflex, int exhaustiveness, Real rmsd_limit){

    dim3 grid(exhaustiveness, exhaustiveness, nflex);
    cal_rmsd_matrix<<<grid, TILE_SIZE>>>(aux_rmsd_matrix_cu, poses_cu, list_flex_topo, exhaustiveness);

    get_pose_energy<<<nflex * exhaustiveness / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
        poses_cu, aux_list_e_cu, nflex, exhaustiveness);
    checkCUDA(cudaDeviceSynchronize());

    // Copy GPU results to CPU
    std::vector<Real> list_e(nflex * exhaustiveness);
    checkCUDA(cudaMemcpy(list_e.data(), aux_list_e_cu, nflex * exhaustiveness * sizeof(Real), cudaMemcpyDeviceToHost));

    std::vector<Real> rmsd_matrix((size_t)nflex * exhaustiveness * exhaustiveness);
    checkCUDA(cudaMemcpy(rmsd_matrix.data(), aux_rmsd_matrix_cu, nflex * exhaustiveness * exhaustiveness * sizeof(Real), cudaMemcpyDeviceToHost));

    // Greedy Leader Selection per flex
    std::vector<int> clustered_pose_inds;
    for (int i = 0; i < nflex; i++){
        int base_e = i * exhaustiveness;
        int base_rmsd = i * exhaustiveness * exhaustiveness;

        std::vector<int> sorted_indices(exhaustiveness);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&](int a, int b) { return list_e[base_e + a] < list_e[base_e + b]; });

        std::vector<int> leaders;
        for (int idx : sorted_indices) {
            bool novel = true;
            for (int leader : leaders) {
                int a = idx, b = leader;
                if (a > b) {
                    std::swap(a, b);
                }
                if (rmsd_matrix[base_rmsd + a * exhaustiveness + b] < rmsd_limit) {
                    novel = false;
                    break;
                }
            }
            if (novel){
                leaders.push_back(idx);
            }
        }

        std::vector<int> list_tmp;
        for (int leader : leaders) {
            int global_idx = base_e + leader;
            clustered_pose_inds.push_back(global_idx);
            list_tmp.push_back(global_idx);
        }
        clustered_pose_inds_list->push_back(list_tmp);
    }

    *out_npose_clustered = clustered_pose_inds.size();
    checkCUDA(
        cudaMemcpy(out_clustered_pose_inds_cu, clustered_pose_inds.data(), clustered_pose_inds.size() * sizeof(int),
            cudaMemcpyHostToDevice));
}
