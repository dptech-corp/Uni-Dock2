//
// Created by Congcong Liu on 24-11-21.
//

#include <iostream>
#include <cooperative_groups/reduce.h>
#include "cluster/cluster.h"
#include "myutils/errors.h"
#include "common.cuh"


namespace cg = cooperative_groups;

__device__ __forceinline__ Real cal_rmsd_pair_warp(const cg::thread_block_tile<TILE_SIZE>& tile,
                                                   const FlexPose* pose1, const FlexPose* pose2, int natom){
    Real rmsd = 0;
    Real tmp = 0;
    for (int i = tile.thread_rank(); i < natom; i += tile.num_threads()){
        tmp = pose1->coords[i * 3] - pose2->coords[i * 3];
        rmsd += tmp * tmp;
        tmp = pose1->coords[i * 3 + 1] - pose2->coords[i * 3 + 1];
        rmsd += tmp * tmp;
        tmp = pose1->coords[i * 3 + 2] - pose2->coords[i * 3 + 2];
        rmsd += tmp * tmp;
    }
    tile.sync();
    rmsd = cg::reduce(tile, rmsd, cg::plus<Real>());

    return sqrt(rmsd / natom);
}


__global__ void cal_rmsd(int* aux_list_cluster, const FlexPose* poses, const int* aux_rmsd_ij,
                         const FlexTopo* list_flex_topo, int npose_per_flex, Real rmsd_limit){
    // one block := one warp := one tile := one pair of poses

    int npair_per_flex = npose_per_flex * (npose_per_flex + 1) / 2;
    int id_flex = blockIdx.x / npair_per_flex;

    // Cooperative group
    auto cta = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(cta);

    int j = aux_rmsd_ij[2 * blockIdx.x];
    int k = aux_rmsd_ij[2 * blockIdx.x + 1];
    tile.sync();
    assert(j < k);

    const FlexPose& pose1 = poses[id_flex * npose_per_flex + j];
    const FlexPose& pose2 = poses[id_flex * npose_per_flex + k];

    Real rmsd = cal_rmsd_pair_warp(tile, &pose1, &pose2, (list_flex_topo + id_flex)->natom);
    if (tile.thread_rank() == 0){
        // record energy
        if (rmsd < rmsd_limit){
            // two poses are deemed as the same
            if (pose1.energy < pose2.energy){
                // pose2 is abandoned
                aux_list_cluster[id_flex * npose_per_flex + k] = 0;
            }
            else{
                // pose1 is left
                aux_list_cluster[id_flex * npose_per_flex + j] = 0;
            }
        }
    }
    tile.sync();
}


__global__ void get_pose_energy(const FlexPose* poses, Real* out_list_e, int nflex, int npose){
    int i_thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_thread < nflex * npose){
        // int i_flex = i_thread / npose;
        out_list_e[i_thread] = poses[i_thread].energy;
    }
}

/**
 * @brief Use a npose*npose matrix to record the merging state between each pair of poses.
 * [NEW VERSION]
 * if aux_cluster_matrix[i][j][j] == 1, then pose[i][j] is left;
 * if aux_cluster_matrix[i][j][j] == 0, then pose[i][j] is abandoned.
 * [OLD VERSION]
 *  if aux_cluster_matrix[i][j][k] == 1, then pose[i][j] is left
 *  if aux_cluster_matrix[i][j][k] == 0, then pose[i][j] is abandoned since it is merged with pose[i][k]
 *  Of course, they two can be both left, which means they are not the same, or j == k.
 *  Then this matrix is used to find the best poses for each flex by counting the number of ZERO in each row.
 * @param out_clustered_pose_inds_cu
 * @param poses_cu
 * @param list_flex_topo
 * @param aux_list_e_cu
 * @param aux_list_cluster_cu
 * @param aux_rmsd_ij_cu
 * @param nflex
 * @param exhaustiveness
 * @param rmsd_limit
 */
void cluster_cu(int* out_clustered_pose_inds_cu, int* out_npose_clustered, std::vector<std::vector<int>>* clustered_pose_inds_list,
                const FlexPose* poses_cu, const FlexTopo* list_flex_topo,
                Real* aux_list_e_cu, int* aux_list_cluster_cu, int* aux_rmsd_ij_cu,
                int nflex, int exhaustiveness, Real rmsd_limit){

    const int block_size = TILE_SIZE; // One block for one tile (for 32, namely one warp per block)
    int npair_per_flex = exhaustiveness * (exhaustiveness - 1) / 2;

    cal_rmsd<<<nflex * npair_per_flex, block_size>>>(aux_list_cluster_cu, poses_cu, aux_rmsd_ij_cu,
                                                     list_flex_topo, exhaustiveness, rmsd_limit);

    get_pose_energy<<<nflex * exhaustiveness / block_size + 1, block_size>>>(
        poses_cu, aux_list_e_cu, nflex, exhaustiveness);
    checkCUDA(cudaDeviceSynchronize());


    // copy from GPU to CPU
    std::vector<Real> list_e(nflex * exhaustiveness);
    checkCUDA(cudaMemcpy(list_e.data(), aux_list_e_cu, nflex * exhaustiveness * sizeof(Real), cudaMemcpyDeviceToHost));
    std::vector<int> list_cluster_states(nflex * exhaustiveness, -1);
    checkCUDA(
        cudaMemcpy(list_cluster_states.data(), aux_list_cluster_cu, nflex * exhaustiveness * sizeof(int),
            cudaMemcpyDeviceToHost));


    std::vector<int> clustered_pose_inds;
    // std::vector<int> sorted_indices(exhaustiveness); // find the best num_modes poses for each flex
    for (int i = 0; i < nflex; i++){
        int tmp = i * exhaustiveness;
        std::vector<int> list_tmp;
        for (int j = 0; j < exhaustiveness; j++){
            if (list_cluster_states[tmp + j] == 1){
                clustered_pose_inds.push_back(tmp + j);
                list_tmp.push_back(tmp+j);
            }
        }
        clustered_pose_inds_list->push_back(list_tmp);
    }

    *out_npose_clustered = clustered_pose_inds.size();
    checkCUDA(
        cudaMemcpy(out_clustered_pose_inds_cu, clustered_pose_inds.data(), clustered_pose_inds.size() * sizeof(int),
            cudaMemcpyHostToDevice));
}
