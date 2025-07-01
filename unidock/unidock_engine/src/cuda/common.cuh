//
// Created by Congcong Liu on 24-10-9.
//

#ifndef COMMON_CUH
#define COMMON_CUH

#include <curand_kernel.h>
#include "myutils/mymath.h"
#include "score/vina.h"
#include <cuda_runtime.h>

extern __device__ __managed__ unsigned int funcCallCount;

// -------------- Constants -----------------
extern __constant__ bool FLAG_CONSTRAINT_DOCK;
extern __constant__ Real BOX_X_HI;
extern __constant__ Real BOX_X_LO;
extern __constant__ Real BOX_Y_HI;
extern __constant__ Real BOX_Y_LO;
extern __constant__ Real BOX_Z_HI;
extern __constant__ Real BOX_Z_LO;
extern __constant__ Real TOR_PREC;
extern __constant__ Real BOX_PREC;

#if true
extern __constant__ Vina Score;
#else
extern __constant__ Gaff2 Score;
#endif

void init_constants(const DockParam& dock_param);


#define TILE_SIZE 32
#define STRIDE_POSE 4
#define STRIDE_G 6



/**
 * @brief Generate an integer in [min, max] on CUDA
 * 
 * @param state CUDA random state
 * @param min Minimum value (int)
 * @param max Maximum value (int)
 * @return An integer in [min, max]
 */
__forceinline__ __device__ int gen_rand_int_within(curandStatePhilox4_32_10_t* state, int min, int max) {
    int range = max - min + 1;
    return min + (curand(state) % range);
}


/**
 * @brief Generate a real number in [min, max]
 * 
 * @param c A random integer within Integer range   
 * @param min Minimum value (Real)
 * @param max Maximum value (Real)
 * @param n Number of divisions. The larger n is, the more accurate the result is.
 * @return A real number in [min, max]
 */
SCOPE_INLINE Real get_real_within_by_int(int c, Real min, Real max, int n=1001) {
    Real v = Real(c % n) * (max - min) / n + min;
    if (v < min){
        return min;
    }
    if(v > max){
        return max;
    }
    return v;
}



/**
 * @brief Generate a real number in [min, max] on CUDA
 * 
 * @param state CUDA random state
 * @param min Minimum value (Real)
 * @param max Maximum value (Real)
 * @return A real number in [min, max]
 */
__forceinline__ __device__ Real gen_rand_real_within(curandStatePhilox4_32_10_t* state, Real min, Real max) {
    return get_real_within_by_int(curand(state), min, max);
}




__forceinline__ __device__ void gen_4_rand_in_sphere(Real *out_rand_4, curandStatePhilox4_32_10_t *state) {
    float4 rand_4;
    while (true) {  // on average, this will have to be run about twice
        rand_4 = curand_uniform4(state);  // ~ U[0,1]
        out_rand_4[0] = (rand_4.x - 0.5) * 2.0; // [-1, 1]
        out_rand_4[1] = (rand_4.y - 0.5) * 2.0;
        out_rand_4[2] = (rand_4.z - 0.5) * 2.0;

        if (cal_norm(out_rand_4) < 1) { // not a specific length, also a randomization on rotation degree
            out_rand_4[3] = rand_4.w;
            return;
        }
    }
}


SCOPE_INLINE Real gyration_radius(const FlexPose* flex_pose, const FlexTopo* flex_topo){
    // compute gyration radius:
    Real d_sq_sum = 0;
    int natom = 0;
    // for each atom
    for (int i = 0; i < flex_topo->natom; i++){
        // only tackle non-H atoms
        if (flex_topo->vn_types[i] != VN_TYPE_H){
            d_sq_sum += dist_sq(flex_pose->coords + 3 * i, flex_pose->center);
            ++natom;
        }
    }
    return natom > 0 ? sqrtf(d_sq_sum / natom) : 0;
}



#endif //COMMON_CUH


