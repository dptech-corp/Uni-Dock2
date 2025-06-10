//
// Created by Congcong Liu on 25-4-17.
//

#ifndef ROTATION_H
#define ROTATION_H


#include "myutils/common.h"
#include "myutils/mymath.h"
#include "myutils/matrix.h"

/**
 * @brief Convert rotation angles to axis-angle representation
 * @snippet test_quaternion.cpp rotvec_to_axis_angle
 * @param out_axis_angle The output axis-angle representation
 * @param rot_vec The 1x3 rotation vector
 */
SCOPE_INLINE void rotvec_to_axis_angle(Real* out_axis_angle, const Real* rot_vec, bool norm=false){
    Real angle = cal_norm(rot_vec);

    if (angle > EPSILON){
        // the real rotation angle
        if (norm){
            out_axis_angle[0] = normalize_angle_2pi(angle);
            if (out_axis_angle[0] > PI){
                out_axis_angle[0] = 2 * PI - out_axis_angle[0];
                angle = -angle; // add a minus sign
            }

        }else{
            out_axis_angle[0] = angle;

        }

        out_axis_angle[1] = rot_vec[0] / angle;
        out_axis_angle[2] = rot_vec[1] / angle;
        out_axis_angle[3] = rot_vec[2] / angle;

    } else{
        out_axis_angle[0] = 0;
        out_axis_angle[1] = 0;
        out_axis_angle[2] = 0;
        out_axis_angle[3] = 0;
    }
}


/**
 * @brief Rotate a vector using Rodrigues' rotation formula
 * @snippet test_quaternion.cpp rotate_vec_by_rodrigues
 * @param out_vec The rotated vector
 * @param axis The rotation axis, a unit vector
 * @param theta The rotation angle
 */
SCOPE_INLINE void rotate_vec_by_rodrigues(Real* out_vec, const Real* axis, Real theta){
    Real tmp[3] = {0};

    Real c = cos(theta);
    Real m = (1 - c) * dot_product(axis, out_vec);
    cross_product(axis, out_vec, tmp);

    out_vec[0] = c * out_vec[0] + m * axis[0] + sin(theta) * tmp[0];
    out_vec[1] = c * out_vec[1] + m * axis[1] + sin(theta) * tmp[1];
    out_vec[2] = c * out_vec[2] + m * axis[2] + sin(theta) * tmp[2];
}



/**
 * @brief Compute gradient of Rotation over the rotation vector.
 * Ref: Bailey, J., Oliveri, A., & Levin, E. (2013). Rigid Body Energy
 * Minimization on Manifolds for Molecular Docking. J Chem Theory
 * Comput., 23(1), 1–7. https://doi.org/10.1021/ct300272j.Rigid
 *
 * @param v Rotation vector (3x1)
 * @param ind 0, 1, 2 for x, y, z
 * @param out_grad: Output gradient, a (3 x 3) matrix
 */
SCOPE_INLINE void cal_grad_of_rot_over_vec(Real* out_grad, const Real* v, int ind){
    Real angle_axis[4] = {0.};
    Real &theta = angle_axis[0];
    Real* v_bar = angle_axis + 1;
    Real e_i[3] = {0.};
    e_i[ind] = 1.;

    rotvec_to_axis_angle(angle_axis, v, true);
    Real s = sin(theta);
    Real c = cos(theta);
    Real v_bar_skew[9] = {0.};
    Real v_bar_skew_sq[9] = {0.};
    skew(v_bar_skew, v_bar);
    skew_sq(v_bar_skew_sq, v_bar);

    for (int j = 0; j < 9; ++j){
        out_grad[j] = 0.;
    }

    // skew-related: matrix summation
    for (int j = 0; j < 9; ++j){
        out_grad[j] += c * v_bar[ind] * v_bar_skew[j] + s * v_bar[ind] * v_bar_skew_sq[j];
    }

    Real s_div_theta = 0.;
    Real one_minus_c_div_theta = 0.;

    if (theta > EPSILON){
        s_div_theta = s / theta;
        one_minus_c_div_theta = (1 - c) / theta;

    } else{
        Real theta_sq = theta * theta;
        s_div_theta = 1 - theta_sq / 6.;
        one_minus_c_div_theta = theta / 2. - theta * theta_sq / 24;
    }

    Real v_s[3] = {0.};
    for (int j = 0; j < 3; ++j){
        v_s[j] = e_i[j] - v_bar[ind] * v_bar[j];
    }
    Real tmp_skew[9] = {0.};
    skew(tmp_skew, v_s);
    Real mat_c[9] = {0.};
    Real mat_tmp[9] = {0.};

    outer_product(e_i, v_bar, mat_tmp);
    for (int j = 0; j < 9; ++j){
        mat_c[j] += mat_tmp[j];
    }
    outer_product(v_bar, e_i, mat_tmp);
    for (int j = 0; j < 9; ++j){
        mat_c[j] += mat_tmp[j] ;
    }
    outer_product(v_bar, v_bar, mat_tmp);
    for (int j = 0; j < 9; ++j){
        mat_c[j] -= 2 * v_bar[ind] * mat_tmp[j] ;
    }

    // final accumulation
    for (int j = 0; j < 9; ++j){
        out_grad[j] += s_div_theta * tmp_skew[j] + one_minus_c_div_theta * mat_c[j];
    }

}



#endif //ROTATION_H
