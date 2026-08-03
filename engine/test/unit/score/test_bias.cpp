//
// Created by Congcong Liu on 25-8-22.
//

#include <catch2/catch_amalgamated.hpp>
#include "score/vina.h"
#include <functional>


TEST_CASE("zalign bias", "[eval_ef_zalign]") {
    Vina vina;
    Real e = -999;
    Real f[3] = {0.};

    Real e_expected = 15.0;
    Real f_expected[3] = {22.5, 0, 0};
    Real r_[3] = {-1, 0, 0}; // bias position is {0, 0, 0} and flex atom at {1, 0, 0}
    Real a_qt = 1.;
    int vn_type = VN_TYPE_C_H;

    e = vina.eval_ef_zalign(r_, a_qt, vn_type, f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(e_expected, 1e-4));
    for (int i = 0; i < 3; i++){
        REQUIRE_THAT(f[i], Catch::Matchers::WithinAbs(f_expected[i], 1e-4));
    }

    //![eval_ef_zalign]
}



