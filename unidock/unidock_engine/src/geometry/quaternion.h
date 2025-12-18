//
// Created by Congcong Liu on 24-10-17.
//

#ifndef QUATERNION_H
#define QUATERNION_H

#include "myutils/mymath.h"
#include "myutils/matrix.h"
#include "myutils/common.h"


/**
 * @brief Conjugate of quaternion
 * @snippet test_quaternion.cpp quaternion_conjugate
 * @param q The quaternion (w, x, y, z)
 * @param out_q The output conjugate of quaternion (w, x, y, z)
 */
SCOPE_INLINE void quaternion_cal_conjugate(Real* out_q, const Real* q){
    out_q[0] = q[0];
    out_q[1] = -q[1];
    out_q[2] = -q[2];
    out_q[3] = -q[3];
}

/**
 * @brief Change quaternion to its Conjugate
 * @snippet test_quaternion.cpp quaternion_conjugate
 * @param q The quaternion (w, x, y, z)
 * @param out_q The output conjugate of quaternion (w, x, y, z)
 */
SCOPE_INLINE void quaternion_conjugate(Real* out_q){
    out_q[0] = out_q[0];
    out_q[1] = -out_q[1];
    out_q[2] = -out_q[2];
    out_q[3] = -out_q[3];
}


/**
 * @brief Multiply two quaternions: q2 x q1
 * @snippet test_quaternion.cpp quaternion_multiply_left
 * @param out_q1 The first quaternion, also the output quaternion (w, x, y, z)
 * @param q2 The second quaternion (w, x, y, z)
 */
SCOPE_INLINE void quaternion_multiply_left(const Real* q2, Real* out_q1){
    Real tmp[4] = {out_q1[0], out_q1[1], out_q1[2], out_q1[3]};
    out_q1[0] = q2[0] * tmp[0] - q2[1] * tmp[1] - q2[2] * tmp[2] - q2[3] * tmp[3];
    out_q1[1] = q2[0] * tmp[1] + q2[1] * tmp[0] + q2[2] * tmp[3] - q2[3] * tmp[2];
    out_q1[2] = q2[0] * tmp[2] - q2[1] * tmp[3] + q2[2] * tmp[0] + q2[3] * tmp[1];
    out_q1[3] = q2[0] * tmp[3] + q2[1] * tmp[2] - q2[2] * tmp[1] + q2[3] * tmp[0];
}

/**
 * @brief Normalize quaternion to unit quaternion
 * @snippet test_quaternion.cpp normalize_quaternion
 * @param out_q The quaternion (w, x, y, z)
 */
SCOPE_INLINE void quaternion_normalize(Real* out_q){
    Real s = out_q[0] * out_q[0] + out_q[1] * out_q[1] + out_q[2] * out_q[2] + out_q[3] * out_q[3];
    // Omit one assert()
    if (fabs(s - 1) < EPSILON){
        return;
    }
    Real a = sqrtf(s);
    for (int i = 0; i < 4; i++){
        out_q[i] /= a;
    }
}

/**
 * @brief Apply two quaternions, first q1, then q2.
 * @snippet test_quaternion.cpp quaternion_increment
 * @param out_q1 The quaternion to be modified (w, x, y, z)
 * @param q2 The quaternion to apply after out_q1 (w, x, y, z)
 */
SCOPE_INLINE void quaternion_increment(Real* out_q1, const Real* q2){
    // q2 x q1, first q1, then q2
    quaternion_multiply_left(q2, out_q1);
}



/**
 * @brief Convert rotation vector to quaternion
 * @snippet test_quaternion.cpp rotvec_to_quaternion
 * @param out_q The output quaternion (w, x, y, z)
 * @param rot_vec The 1x3 rotation vector
 */
SCOPE_INLINE void rotvec_to_quaternion(Real* out_q, const Real* rot_vec){
    // the real rotation angle
    Real theta = cal_norm(rot_vec);
    if (theta > EPSILON){
        Real axis[3] = {rot_vec[0] / theta, rot_vec[1] / theta, rot_vec[2] / theta};
        Real s = sin(theta / 2);
        out_q[0] = cos(theta / 2);
        out_q[1] = s * axis[0];
        out_q[2] = s * axis[1];
        out_q[3] = s * axis[2];
    }
    else{
        out_q[0] = 1;
        out_q[1] = 0;
        out_q[2] = 0;
        out_q[3] = 0;
    }
}

/**
 * @brief Convert unit quaternion to rotation vector
 * @snippet test_quaternion.cpp quaternion_to_rotvec
 * @param q The original unit quaternion (w, x, y, z)
 * @param out_v The 1x3 rotation vector
 */
SCOPE_INLINE void quaternion_to_rotvec(Real* out_v, const Real* q){
    // the angle
    Real theta = 2 * acos(q[0]);
    Real s = sin(theta / 2);
    if (s > EPSILON){
        Real theta_div_s = theta / s;
        out_v[0] = q[1] * theta_div_s;
        out_v[1] = q[2] * theta_div_s;
        out_v[2] = q[3] * theta_div_s;
    }
    else{
        out_v[0] = 0;
        out_v[1] = 0;
        out_v[2] = 0;
    }
}



/**
 * @brief Convert quaternion to rotation matrix
 * @snippet test_quaternion.cpp quaternion_to_matrix
 * @param q The quaternion (w, x, y, z)
 * @param out_mat The output rotation matrix, 3x3
 */
SCOPE_INLINE void quaternion_to_matrix(const Real* q, Real* out_mat){
    /* Omit assert(quaternion_is_normalized(q)); */
    const Real a = q[0];
    const Real b = q[1];
    const Real c = q[2];
    const Real d = q[3];

    const Real aa = a * a;
    const Real ab = a * b;
    const Real ac = a * c;
    const Real ad = a * d;
    const Real bb = b * b;
    const Real bc = b * c;
    const Real bd = b * d;
    const Real cc = c * c;
    const Real cd = c * d;
    const Real dd = d * d;

    /* Omit assert(eq(aa + bb + cc + dd, 1)); */
    Real tmp[9]; // matrix
    init_3x3_mat(tmp, 0); /* matrix_d with fixed dimension 3(here we treate this as
                          a regular matrix_d(not triangular matrix_d!)) */

    mat_set_element(tmp, 3, 0, 0, (aa + bb - cc - dd));
    mat_set_element(tmp, 3, 0, 1, 2 * (-ad + bc));
    mat_set_element(tmp, 3, 0, 2, 2 * (ac + bd));
    mat_set_element(tmp, 3, 1, 0, 2 * (ad + bc));
    mat_set_element(tmp, 3, 1, 1, (aa - bb + cc - dd));
    mat_set_element(tmp, 3, 1, 2, 2 * (-ab + cd));
    mat_set_element(tmp, 3, 2, 0, 2 * (-ac + bd));
    mat_set_element(tmp, 3, 2, 1, 2 * (ab + cd));
    mat_set_element(tmp, 3, 2, 2, (aa - bb - cc + dd));

    for (int i = 0; i < 9; i++){
        out_mat[i] = tmp[i];
    }
}


/**
 * @brief Rotate a vector by a quaternion
 * @snippet test_quaternion.cpp rotate_vec_by_quaternion
 * @param out_vec The output rotated vector
 * @param q The quaternion (w, x, y, z)
 */
SCOPE_INLINE void rotate_vec_by_quaternion(Real* out_vec, const Real* q){
    Real q_c[4] = {0};
    Real vec4[4] = {0, out_vec[0], out_vec[1], out_vec[2]};

    quaternion_multiply_left(q, vec4);
    quaternion_cal_conjugate(q_c, q);
    quaternion_multiply_left(vec4, q_c);

    out_vec[0] = q_c[1];
    out_vec[1] = q_c[2];
    out_vec[2] = q_c[3];
}



#endif //QUATERNION_H
