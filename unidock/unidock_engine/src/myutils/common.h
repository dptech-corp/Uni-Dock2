//
// Created by Congcong Liu on 24-9-20.
//

#ifndef COMMON_H
#define COMMON_H

#include <limits>
#include <cuda_runtime.h>


// ------------------------------- CPU or CUDA -------------------------------
const Real EPSILON_c = std::numeric_limits<Real>::epsilon();

#ifdef __CUDACC__
    #define SCOPE_CU __host__ __device__
    #define SCOPE_INLINE __host__ __device__ __forceinline__
    __constant__ const Real EPSILON_cu = std::numeric_limits<Real>::epsilon();

#else
    #define SCOPE_CU
    #define SCOPE_INLINE __attribute__((always_inline)) inline
#endif

SCOPE_INLINE Real getEpsilon() {
#ifdef __CUDACC__
        return EPSILON_cu; // GPU
#else
        return EPSILON_c; // CPU
#endif
}

#define EPSILON getEpsilon()


// ------------------------------- CPU -------------------------------
template <typename T>
SCOPE_INLINE T square(T x) {
    return x * x;
}



// ------------------------------- CPU & CUDA -------------------------------
#define PI Real(3.1415927)
#define PHI 1.4142136
#define PSI 1.5337512

constexpr Real DEG2RAD = PI / 180;
constexpr Real RAD2DEG = 180 / PI;

// ------------------------------- DEBUG -------------------------------
#ifdef DEBUG
#define DPrint1(fmt, ...) if(tile.thread_rank() == 0) {printf(fmt, __VA_ARGS__);}
#define DPrint(fmt, ...) printf(fmt, __VA_ARGS__)
#define DPrintCPU(fmt, ...) printf(fmt, __VA_ARGS__)
#define DExec(stmt) if(tile.thread_rank() == 0){stmt}

// #define DPrint1(fmt, ...) if((blockIdx.x == 0) && (tile.thread_rank() == 0)) {printf("[CUDA info]");printf(fmt, __VA_ARGS__);}
// #define DPrint(fmt, ...) if(tile.thread_rank() == 0) {printf("[CUDA info]");printf(fmt, __VA_ARGS__);}

#else

#define DPrint1(fmt, ...)
#define DPrint(fmt, ...)
#define DPrintCPU(fmt, ...)
#define DExec(stmt)

#endif





#endif //COMMON_H
