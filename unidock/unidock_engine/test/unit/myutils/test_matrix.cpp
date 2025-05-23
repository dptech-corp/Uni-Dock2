//
// Created by Congcong Liu on 25-4-17.
//

#include <catch2/catch_amalgamated.hpp>
#include "myutils/matrix.h"

TEST_CASE("init_3x3_mat initializes a 3x3 matrix with a given value", "[init_3x3_mat]") {
    Real m[9];
    init_3x3_mat(m, 1.0f);
    for (int i = 0; i < 9; ++i) {
        REQUIRE_THAT(m[i], Catch::Matchers::WithinAbs(1.0f, 1e-4));
    }
    //![init_3x3_mat]
}

TEST_CASE("mat_set_element sets an element of a 3x3 matrix", "[mat_set_element]") {
    Real m[9];
    mat_set_element(m, 3, 1, 2, 3.0f);
    REQUIRE_THAT(m[1 + 2 * 3], Catch::Matchers::WithinAbs(3.0f, 1e-4));
    //![mat_set_element]
}

TEST_CASE("uptri_mat_index calculates the sequential index for a triangular matrix", "[uptri_mat_index]") {
    REQUIRE(uptri_mat_index(0, 0) == 0);
    REQUIRE(uptri_mat_index(1, 1) == 2);
    REQUIRE(uptri_mat_index(0, 1) == 1);
    //![uptri_mat_index]
}

TEST_CASE("tri_mat_index calculates the sequential index for a triangular matrix, allowing i <= j or i >= j", "[tri_mat_index]") {
    REQUIRE(tri_mat_index(0, 0) == 0);
    REQUIRE(tri_mat_index(1, 0) == 1);
    REQUIRE(tri_mat_index(1, 1) == 2);
    REQUIRE(tri_mat_index(0, 1) == 1);
    REQUIRE(tri_mat_index(2, 0) == 3);
    REQUIRE(tri_mat_index(2, 1) == 4);
    REQUIRE(tri_mat_index(2, 2) == 5);
    //![tri_mat_index]
}

