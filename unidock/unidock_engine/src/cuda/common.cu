//
// Created by Congcong Liu on 24-12-12.
//
#include "common.cuh"

__device__ __managed__ unsigned int funcCallCount = 0;


__constant__ bool FLAG_FIX_CORE = false;
__constant__ BiasType BIAS_TYPE = BT_POS;
__constant__ Real BIAS_K = 0.1;

__constant__ Box CU_BOX;


#if true
__constant__ Vina Score;
#else
__constant__ Gaff2 Score;
#endif


void init_constants(const DockParam& dock_param){
    //======================= constants ======================
    checkCUDA(cudaMemcpyToSymbol(BIAS_TYPE, &dock_param.bias_type, sizeof(BiasType)));
    checkCUDA(cudaMemcpyToSymbol(BIAS_K, &dock_param.bias_k, sizeof(Real)));
    checkCUDA(cudaMemcpyToSymbol(FLAG_FIX_CORE, &dock_param.constraint_docking, sizeof(bool)));
    checkCUDA(cudaMemcpyToSymbol(CU_BOX, &dock_param.box, sizeof(dock_param.box), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaDeviceSynchronize());// assure that memcpy is finished
}