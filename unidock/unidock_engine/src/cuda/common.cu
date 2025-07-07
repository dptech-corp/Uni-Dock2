//
// Created by Congcong Liu on 24-12-12.
//
#include "common.cuh"

__device__ __managed__ unsigned int funcCallCount = 0;


__constant__ bool FLAG_CONSTRAINT_DOCK = false;
__constant__ Real BOX_X_HI = 30;
__constant__ Real BOX_X_LO = -30;
__constant__ Real BOX_Y_HI = 30;
__constant__ Real BOX_Y_LO = -30;
__constant__ Real BOX_Z_HI = 30;
__constant__ Real BOX_Z_LO = -30;
__constant__ Real TOR_PREC = 0.3;
__constant__ Real BOX_PREC = 1.0;
__constant__ Real PENALTY_SLOPE = 1e6;


#if true
__constant__ Vina Score;
#else
__constant__ Gaff2 Score;
#endif


void init_constants(const DockParam& dock_param){
    //======================= constants ======================
    checkCUDA(cudaMemcpyToSymbol(FLAG_CONSTRAINT_DOCK, &dock_param.constraint_docking, sizeof(bool)));
    checkCUDA(cudaMemcpyToSymbol(BOX_X_HI, &dock_param.box.x_hi, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(BOX_X_LO, &dock_param.box.x_lo, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(BOX_Y_HI, &dock_param.box.y_hi, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(BOX_Y_LO, &dock_param.box.y_lo, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(BOX_Z_HI, &dock_param.box.z_hi, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(BOX_Z_LO, &dock_param.box.z_lo, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(TOR_PREC, &dock_param.tor_prec, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpyToSymbol(BOX_PREC, &dock_param.tor_prec, sizeof(Real), 0, cudaMemcpyHostToDevice));
    checkCUDA(cudaDeviceSynchronize());// assure that memcpy is finished
}