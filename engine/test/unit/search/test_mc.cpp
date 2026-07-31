//
// Created by lccdp on 24-9-3.
//

#include <catch2/catch_amalgamated.hpp>
#include "search/mc.h"


//![test_mc1]
TEST_CASE("test_mc1", "[mc]"){
    int t = 0;
    REQUIRE(t == 0);
}

//![test_mc1]

//![test_mc2]
TEST_CASE("test_mc2", "[mc]"){
    double t = 0;
    REQUIRE(t == 0);
    REQUIRE(0 == Catch::Approx(t).epsilon(1e-6));
}
//![test_mc2]


//![test_mc3]
TEST_CASE("test_mc3", "[mc]"){
    double t = 0;
    REQUIRE(t == 0);
    REQUIRE(0 == Catch::Approx(t).epsilon(1e-6));
}
//![test_mc3]