//
// Created by Congcong Liu on 24-12-12.
//
#include "common.cuh"

__device__ __managed__ unsigned int funcCallCount = 0;


__constant__ bool FLAG_CONSTRAINT_DOCK = false;
__constant__ Box CU_BOX;


#if true
__constant__ Vina Score;
#else
__constant__ Gaff2 Score;
#endif


void init_constants(const DockParam& dock_param){
    //======================= constants ======================
    checkCUDA(cudaMemcpyToSymbol(FLAG_CONSTRAINT_DOCK, &dock_param.constraint_docking, sizeof(bool)));
    checkCUDA(cudaMemcpyToSymbol(CU_BOX, &dock_param.box, sizeof(dock_param.box), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaDeviceSynchronize());// assure that memcpy is finished
}