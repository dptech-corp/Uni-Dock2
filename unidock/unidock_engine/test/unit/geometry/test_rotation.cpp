//
// Created by Congcong Liu on 24-10-17.
//

#include <catch2/catch_amalgamated.hpp>
#include "geometry/rotation.h"


TEST_CASE("rotate_vec_by_rodrigues rotates a vector by a quaternion", "[rotate_vec_by_rodrigues]"){
    // test 1: rotate a vector by 90 degrees around the z axis
    Real vec[] = {1.0f, 0.0f, 0.0f};
    Real expected[] = {0.0f, 1.0f, 0.0f};
    Real axis[] = {0.0f, 0.0f, 1.0f};
    Real angle = M_PI / 2;

    rotate_vec_by_rodrigues(vec, axis, angle);
    for (int i = 0; i < 3; ++i){
        REQUIRE_THAT(vec[i], Catch::Matchers::WithinAbs(expected[i], 1e-4));
    }
    std::array<Real, 3> v = {1.0f, 0.0f, 0.0f};
    rotate_vec_by_rodrigues(v.data(), axis,  - 7 * PI / 2 );
    for (int i = 0; i < 3; ++i){
        REQUIRE_THAT(v[i], Catch::Matchers::WithinAbs(expected[i], 1e-4));
    }


    // test 2: rotate a vector by 45 degrees around the x axis
    Real vec2[] = {0.0f, 1.0f, 0.0f};
    Real expected2[] = {0.0f, 0.7071f, 0.7071f};
    Real axis2[] = {1.0f, 0.0f, 0.0f};
    Real angle2 = M_PI / 4;

    rotate_vec_by_rodrigues(vec2, axis2, angle2);
    for (int i = 0; i < 3; ++i){
        REQUIRE_THAT(vec2[i], Catch::Matchers::WithinAbs(expected2[i], 1e-4));
    }

    // corner case: rotate a zero vector by 90 degrees around the z axis
    Real vec3[] = {0.0f, 0.0f, 0.0f};
    Real expected3[] = {0.0f, 0.0f, 0.0f};
    Real axis3[] = {0.0f, 0.0f, 1.0f};
    Real angle3 = M_PI / 2;

    rotate_vec_by_rodrigues(vec3, axis3, angle3);
    for (int i = 0; i < 3; ++i){
        REQUIRE_THAT(vec3[i], Catch::Matchers::WithinAbs(expected3[i], 1e-4));
    }
    //! [rotate_vec_by_rodrigues]
}



TEST_CASE("cal_grad_of_rot_over_vec compute grad of Rotation over a rot vec", "[cal_grad_of_rot_over_vec]"){
    Real out_grad[9] = {0.};
    Real v[3] = {12.3, -7.3, 8.2};

    std::array<std::array<Real, 9>, 3> grad_python = {{
        // x
        {{
            0.30110348,  0.42657892,  0.02066039,
            -0.06989328, 0.39102033,  0.56450775,
            -0.42132096, -0.26542778, 0.35617097
        }},
        // y
        {{
            -0.08676937, -0.20300266, -0.02423069,
            0.09165157,  -0.3240033,  -0.2581077,
            0.32331066,   0.18387365, -0.21138602
        }},
        // z
        {{
            0.09746696,  0.35764433,  0.05680221,
            -0.05856436, 0.26068022,  0.30196226,
            -0.23785202, -0.19450994, 0.340716
        }}
    }};

    for (int ind = 0; ind < 3; ++ind){
        cal_grad_of_rot_over_vec(out_grad, v, ind);
        for (int i = 0; i < 9; ++i){
            REQUIRE_THAT(out_grad[i], Catch::Matchers::WithinAbs(grad_python[ind][i], 1e-4));
        }
    }
    //! [cal_grad_of_rot_over_vec]
}


void update_dihe(Real dihedral, Real dihe_incre_raw, Real *out_dihe, Real *out_dihe_incre, bool norm=true){
    Real incre_tmp = dihe_incre_raw;
    Real range[2] = {-PI, PI};

    if (norm){
        incre_tmp = normalize_angle(incre_tmp);
    }

    Real tmp = normalize_angle(dihedral + incre_tmp);
    *out_dihe = clamp_by_ranges(tmp, range ,1);
    *out_dihe_incre = *out_dihe - dihedral;
}




TEST_CASE("tmp", "tmp me"){
    const Real LIMIT = 1e-6;

    //todo:  check boundary value -PI
    std::array<Real, 3> r0 = {1, 2, 3};

    std::array<Real, 3> axis_unit = {1, 0, 0};

    for (int j = -10; j < 10; ++j){
        Real dihedral = j * 0.2 * PI;

        for (int i = -100; i < 1000; ++i){
            Real dihe_incre_raw = i * 0.135792 * PI;
            Real s = sin(dihe_incre_raw);
            Real s_norm = sin(normalize_angle(dihe_incre_raw));
            REQUIRE_THAT(s, Catch::Matchers::WithinAbs(s_norm, LIMIT));

            return;

            Real dihe_update = -999, dihe_update_norm = -999;
            Real dihe_incre = -999, dihe_incre_norm = -999;
            update_dihe(dihedral, dihe_incre_raw, &dihe_update, &dihe_incre, false);
            update_dihe(dihedral, dihe_incre_raw, &dihe_update_norm, &dihe_incre_norm, true);
            std::array<Real, 3> r = r0;
            std::array<Real, 3> r_norm = r0;

            for (int k = 0; k < 100; ++k){
                update_dihe(dihe_update, dihe_incre_raw, &dihe_update, &dihe_incre, false);
                update_dihe(dihe_update_norm, dihe_incre_raw, &dihe_update_norm, &dihe_incre_norm, true);
                // REQUIRE_THAT(dihe_incre, Catch::Matchers::WithinAbs(dihe_incre_norm, LIMIT));

                rotate_vec_by_rodrigues(r.data(), axis_unit.data(), dihe_incre);
                rotate_vec_by_rodrigues(r_norm.data(), axis_unit.data(), dihe_incre_norm);
                REQUIRE_THAT(dist(r.data(), r_norm.data()), Catch::Matchers::WithinAbs(0, LIMIT));

                // REQUIRE_THAT(sin(dihe_incre), Catch::Matchers::WithinAbs(sin(dihe_incre_norm), LIMIT));
                // REQUIRE_THAT(cos(dihe_incre), Catch::Matchers::WithinAbs(cos(dihe_incre_norm), LIMIT));
            }
        }
    }
}



