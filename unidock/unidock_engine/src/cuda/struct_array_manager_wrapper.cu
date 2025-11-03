//
// Created for testing StructArrayManager template
// Contains test structures and kernels for unit tests
//

#include "cuda/struct_array_manager.cuh"
#include "myutils/errors.h"

//==========================================================
// Test structures and kernels
//==========================================================

// Test structure
struct MyBox {
    int*    ids;      // Array of int per object
    float*  values;   // Array of float per object
    float   scale;    // Regular member
};

__global__ void scale_values_kernel(MyBox* objs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    
    // Assume each object's values array length is (3 + idx)
    int len = 3 + idx;
    for (int i = 0; i < len; ++i) {
        objs[idx].values[i] *= objs[idx].scale;
    }
}

__global__ void modify_ids_kernel(MyBox* objs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    
    // Assume each object's ids array length is (2 + idx)
    int len = 2 + idx;
    for (int i = 0; i < len; ++i) {
        objs[idx].ids[i] += 100;
    }
}

//==========================================================
// Wrapper functions for testing
//==========================================================

void scale_values_gpu(MyBox* objs_device, int n) {
    scale_values_kernel<<<1, 32>>>(objs_device, n);
    checkCUDA(cudaDeviceSynchronize());
}

void modify_ids_gpu(MyBox* objs_device, int n) {
    modify_ids_kernel<<<1, 32>>>(objs_device, n);
    checkCUDA(cudaDeviceSynchronize());
}

