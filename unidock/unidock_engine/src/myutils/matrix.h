//
// Created by Congcong Liu on 25-4-17.
//

#ifndef MATRIX_H
#define MATRIX_H
#include "myutils/common.h"


//------------------------------------------------------
//------------------- Matrix Operations ----------------
//------------------------------------------------------


/**
 * @brief Generate a skew symmetric matrix (for cross product) from a vector
 * @snippet test_math.cpp skew
 * @param out_m The matrix to be initialized
 */
SCOPE_INLINE void skew(Real* out_m, const Real* v){
    out_m[0] = 0;     out_m[1] = -v[2]; out_m[2] = v[1];
    out_m[3] = v[2];  out_m[4] = 0;     out_m[5] = -v[0];
    out_m[6] = -v[1]; out_m[7] = v[0];  out_m[8] = 0;
}

/**
 * @brief Generate a skew symmetric matrix (for cross product) from a vector
 * @snippet test_math.cpp skew_sq
 * @param out_m The matrix to be initialized
 */
SCOPE_INLINE void skew_sq(Real* out_m, const Real* v){
    Real v0v1 = v[0] * v[1];
    Real v1v2 = v[1] * v[2];
    Real v0v2 = v[0] * v[2];
    Real v0_sq = v[0] * v[0];
    Real v1_sq = v[1] * v[1];
    Real v2_sq = v[2] * v[2];

    out_m[0] = -v1_sq - v2_sq; out_m[1] = v0v1;           out_m[2] = v0v2;
    out_m[3] = v0v1;           out_m[4] = -v0_sq - v2_sq; out_m[5] = v1v2;
    out_m[6] = v0v2;           out_m[7] = v1v2;           out_m[8] = -v0_sq - v1_sq;
}

/**
 * @brief Compute the Frobenius dot product of two 3x3 matrix
 * @snippet test/unit/myutils/test_matrix.cpp frobenius_product
 * @param a The first matrix
 * @param b The second matrix
 */
template <typename T1, typename T2>
SCOPE_INLINE Real frobenius_product(const T1* a, const T2* b){
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] +
           a[3] * b[3] + a[4] * b[4] + a[5] * b[5] +
           a[6] * b[6] + a[7] * b[7] + a[8] * b[8];
}

/**
 * @brief Initialize a 3x3 matrix with a given value
 * @snippet test_math.cpp init_3x3_mat
 * @param out_m The matrix to be initialized
 * @param fill_value The value to fill the matrix with
 */
SCOPE_INLINE void init_3x3_mat(Real* out_m, Real fill_value){
    // fixed to 3x3 matrix_d
    for (int i = 0; i < 9; i++){
        out_m[i] = fill_value;
    }
}


/**
 * @brief Set an element of a matrix
 * @snippet test_math.cpp mat_set_element
 * @param out_m The matrix to be set
 * @param dim The dimension of the matrix
 * @param i The row index
 * @param j The column index
 * @param fill_value The value to set the element to
 */
SCOPE_INLINE void mat_set_element(Real* out_m, int dim, int i, int j, Real fill_value){
    out_m[i + j * dim] = fill_value;
}


/**
 * @brief Calculate the sequential index for an upper triangular matrix (i <= j)
 * @snippet test_math.cpp uptri_mat_index
 * @param i The row index
 * @param j The column index
 * @return The index of the element in the sequential triangular matrix
 */
SCOPE_INLINE int uptri_mat_index(int i, int j){
    // assert(i <= j);
    // from i_col, i_row to ind in sequence
    return i + j * (j + 1) / 2;
}

/**
 * @brief Calculate the sequential index for a triangular matrix, including upper- & lower-
 * @snippet test_math.cpp tri_mat_index
 * @param i The row index
 * @param j The column index
 * @return The index of the element in the sequential triangular matrix
 */
SCOPE_INLINE int tri_mat_index(int i, int j){
    return (i < j) ? uptri_mat_index(i, j) : uptri_mat_index(j, i);
}


//------------------------------------------------------
//------------------ Matrix and Vector Operations ------
//------------------------------------------------------

SCOPE_INLINE void mat_multiply_vec(Real out_v[3], const Real m[9], const Real v[3]){
    out_v[0] = m[0] * v[0] + m[1] * v[0] + m[2] * v[0];
    out_v[1] = m[0] * v[1] + m[1] * v[1] + m[2] * v[1];
    out_v[2] = m[0] * v[2] + m[1] * v[2] + m[2] * v[2];
}


#endif //MATRIX_H
