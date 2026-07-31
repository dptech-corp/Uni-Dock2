//
// Created by Congcong Liu on 24-10-16.
//

#include <catch2/catch_amalgamated.hpp>
#include "myutils/mymath.h"

TEST_CASE("normalize_angle normalizes an angle to the range [-π, π)", "[normalize_angle]") {
    Real margin = 1e-4;
    REQUIRE_THAT(normalize_angle(0.0f), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle(PI), Catch::Matchers::WithinAbs(PI, margin)); // 注意，PI
    REQUIRE_THAT(normalize_angle(-PI), Catch::Matchers::WithinAbs(-PI, margin));
    REQUIRE_THAT(normalize_angle(2 * PI), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle(-2 * PI), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle(3 * PI), Catch::Matchers::WithinAbs(PI, margin));
    REQUIRE_THAT(normalize_angle(-3 * PI), Catch::Matchers::WithinAbs(-PI, margin));
    //![normalize_angle]
}

TEST_CASE("normalize_angle normalizes an angle to the range [0, 2π)", "[normalize_angle]") {
    Real margin = 1e-4;
    REQUIRE_THAT(normalize_angle_2pi(0.0f), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle_2pi(PI), Catch::Matchers::WithinAbs(PI, margin)); // 注意，PI 会被转换为 -PI
    REQUIRE_THAT(normalize_angle_2pi(-PI), Catch::Matchers::WithinAbs(PI, margin));
    REQUIRE_THAT(normalize_angle_2pi(2 * PI), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle_2pi(-2 * PI), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle_2pi(0.3 + 13 * PI), Catch::Matchers::WithinAbs(3.441592, margin));
    REQUIRE_THAT(normalize_angle_2pi(0.3 - 13 * PI), Catch::Matchers::WithinAbs(3.441592, margin));
    //![normalize_angle]
}


TEST_CASE("ang_to_rad converts an angle in degree to radian", "[ang_to_rad]") {
    REQUIRE_THAT(ang_to_rad(0), Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(ang_to_rad(180), Catch::Matchers::WithinAbs(PI, 1e-4));
    REQUIRE_THAT(ang_to_rad(90), Catch::Matchers::WithinAbs(PI / 2, 1e-4));
    REQUIRE_THAT(ang_to_rad(-180), Catch::Matchers::WithinAbs(-PI, 1e-4));
    //![ang_to_rad]
}


TEST_CASE("get_radian_in_ranges randomly selects a value within a given range", "[get_radian_in_ranges]") {
    Real ranges[] = {-180, -160, 20, 90, 110, 150};
    int num_range = 3;

    Real rand[] = {0.54f, 0.1f};
    REQUIRE_THAT(get_radian_in_ranges(ranges, num_range, rand), Catch::Matchers::WithinAbs(27, 1e-4));

    Real rand2[] = {0.89f, 1e-6};
    REQUIRE_THAT(get_radian_in_ranges(ranges, num_range, rand2), Catch::Matchers::WithinAbs(110, 1e-4));

    Real rand3[] = {0.21f, 1.0f}  ;
    REQUIRE_THAT(get_radian_in_ranges(ranges, num_range, rand3), Catch::Matchers::WithinAbs(-160, 1e-4));
    //![get_radian_in_ranges]
}


TEST_CASE("get_real_within_by_int", "[get_real_within_by_int]") {
    // default n = 1001
    REQUIRE_THAT(get_real_within_by_int(0, 0, 10), Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(get_real_within_by_int(11, 0, 10, 1001), Catch::Matchers::WithinAbs(11.0/1001 * 10, 1e-4));
    REQUIRE_THAT(get_real_within_by_int(1002, 0, 10, 1001), Catch::Matchers::WithinAbs(1.0/1001 * 10, 1e-4));
    // x
    REQUIRE_THAT(get_real_within_by_int(633093634, -30.5, -5.5), Catch::Matchers::WithinAbs(-23.2419, 1e-4));
    REQUIRE_THAT(get_real_within_by_int(633, -PI, PI), Catch::Matchers::WithinAbs(-0.5067, 1e-4));

    //![get_real_within_by_int]
}

TEST_CASE("get_real_within", "[get_real_within]") {
    // left, mid, right
    REQUIRE_THAT(get_real_within(0.0, M_PI, 10.0), Catch::Matchers::WithinAbs(M_PI, 1e-4));
    REQUIRE_THAT(get_real_within(0.5, 5.0, 10.0), Catch::Matchers::WithinAbs(7.5, 1e-4));
    REQUIRE_THAT(get_real_within(0.99999, -3.8, -1.02), Catch::Matchers::WithinAbs(-1.02, 1e-4));

    // min == max no error
    REQUIRE_THAT(get_real_within(0.1234, 2.5, 2.5),
                Catch::Matchers::WithinAbs(2.5, 1e-4));
    //![get_real_within]
}


TEST_CASE("compute gaussian", "[gaussian]") {
    Real e, f;
    e = gaussian(0, 0, 3, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));

    e = gaussian(1, 1, 3, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));

    e = gaussian(0, 1, 4, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0.939413071f, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0.117426634f, 1e-4));
}


TEST_CASE("computes the cross product of two 3D vectors", "[cross_product]") {
    Real vec1[] = {1.0f, 2.0f, 3.0f};
    Real vec2[] = {4.0f, 5.0f, 6.0f};
    Real result[3];
    cross_product(vec1, vec2, result);
    REQUIRE_THAT(result[0], Catch::Matchers::WithinAbs(-3.0f, 1e-4));
    REQUIRE_THAT(result[1], Catch::Matchers::WithinAbs(6.0f, 1e-4));
    REQUIRE_THAT(result[2], Catch::Matchers::WithinAbs(-3.0f, 1e-4));


    double vec3[] = {1.0, 0.0, 0.0};
    double vec4[] = {0.0, 1.0, 0.0};
    double result2[3];
    cross_product(vec3, vec4, result2);
    REQUIRE_THAT(result2[0], Catch::Matchers::WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(result2[1], Catch::Matchers::WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(result2[2], Catch::Matchers::WithinAbs(1.0, 1e-4));

    int vec5[] = {1, 0, 0};
    int vec6[] = {0, 1, 0};
    int result3[3];
    cross_product(vec5, vec6, result3);
    REQUIRE(result3[0] == 0);
    REQUIRE(result3[1] == 0);
    REQUIRE(result3[2] == 1);   
    //![cross_product]
}


TEST_CASE("computes the outer product of two 3D vectors", "[outer_product]") {
    Real vec1[] = {1.0f, 2.0f, 3.0f};
    Real vec2[] = {4.0f, 5.0f, 6.0f};
    Real expected[9] = {4, 5, 6, 8, 10, 12, 12, 15, 18};
    Real result[9];
    outer_product(vec1, vec2, result);
    for (int i = 0; i < 9; ++i){
        REQUIRE_THAT(result[i], Catch::Matchers::WithinAbs(expected[i], 1e-4));
    }
    //![outer_product]
}


TEST_CASE("compute dot product", "[dot_product]") {
    Real vec1[] = {1.0f, 2.0f, 3.0f};
    Real vec2[] = {4.0f, 5.0f, 6.0f};
    REQUIRE_THAT(dot_product(vec1, vec2), Catch::Matchers::WithinAbs(32.0f, 1e-4));
    //![dot_product]
}

TEST_CASE("cal_norm computes the Euclidean norm of a 3D vector", "[cal_norm]") {
    Real margin = 1e-4;

    Real vec1[] = {3.0f, 4.0f, 0.0f};
    REQUIRE_THAT(cal_norm(vec1), Catch::Matchers::WithinAbs(5.0f, margin));

    Real vec2[] = {1.0f, 2.0f, 2.0f};
    REQUIRE_THAT(cal_norm(vec2), Catch::Matchers::WithinAbs(3.0f, margin));

    Real vec3[] = {0.0f, 0.0f, 0.0f};
    REQUIRE_THAT(cal_norm(vec3), Catch::Matchers::WithinAbs(0.0f, margin));
    //![cal_norm]
}

TEST_CASE("dist2 computes the squared Euclidean distance between two 3D points", "[dist2]") {
    Real margin = 1e-4;

    Real point1[] = {1.0f, 2.0f, 3.0f};
    Real point2[] = {4.0f, 6.0f, 8.0f};
    REQUIRE_THAT(dist_sq(point1, point2), Catch::Matchers::WithinAbs(50.0f, margin));

    Real point3[] = {0.0f, 0.0f, 0.0f};
    Real point4[] = {0.0f, 0.0f, 0.0f};
    REQUIRE_THAT(dist_sq(point3, point4), Catch::Matchers::WithinAbs(0.0f, margin));

    Real point5[] = {1.0f, 1.0f, 1.0f};
    Real point6[] = {2.0f, 2.0f, 2.0f};
    REQUIRE_THAT(dist_sq(point5, point6), Catch::Matchers::WithinAbs(3.0f, margin));
    //![dist2]
}



