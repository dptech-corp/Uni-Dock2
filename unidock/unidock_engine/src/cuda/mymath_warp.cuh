//
// Created by Congcong Liu on 24-10-22.
//

#ifndef MYMATH_WARP_H
#define MYMATH_WARP_H

#include <curand_kernel.h>



__device__ __forceinline__ void init_tri_mat_warp(const cooperative_groups::thread_block_tile<TILE_SIZE> &tile,
                                                 Real *m, int dim, Real fill_data) {
    // no padding, since the size is variable for each ligand
    for (int i = tile.thread_rank(); i < dim * (dim + 1) / 2; i += tile.num_threads()) {
        m[i] = fill_data;
    }
    tile.sync();
}


__device__ __forceinline__ void set_tri_mat_diagonal_warp(const cooperative_groups::thread_block_tile<TILE_SIZE> &tile,
                                                    Real *m, int dim, Real fill_data) {
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()) {
        m[i + i * (i + 1) / 2] = fill_data;
    }
    tile.sync();
}











#endif //MYMATH_WARP_H
