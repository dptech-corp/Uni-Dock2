//
// Created by Congcong Liu on 24-11-12.
//


#include <catch2/catch_amalgamated.hpp>
#include "score/vina.h"
#include <functional>

template<typename F>
Real cal_f_by_fd(F func, Real d, Real delta=1e-3){
    Real tmp = 0;
    double e1 = func(d + delta, &tmp);
    double e2 = func(d - delta, &tmp);
    return (e1 - e2) / delta / 2;
}

TEST_CASE("vina computes gaussian1", "[vina_gaussian1]") {
    Vina vina;
    Real e;
    Real f;

    e = vina.vina_gaussian1(-1, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0.018315, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0.146525, 1e-4));

    e = vina.vina_gaussian1(0, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(1.0, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0.0, 1e-4));

    e = vina.vina_gaussian1(1, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0.018315, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(-0.146525, 1e-4));

    e = vina.vina_gaussian1(1.5, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0.000123409804087, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(-0.00148091764904, 1e-4));
    //![vina_gaussian1]
}

TEST_CASE("vina computes gaussian2", "[vina_gaussian2]") {
    Vina vina;
    Real e;
    Real f;
    e = vina.vina_gaussian2(0, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0.105399224562, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0.158098836843, 1e-4));

    e = vina.vina_gaussian2(3, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));

    e = vina.vina_gaussian2(6, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0.105399224562, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(-0.158098836843, 1e-4));
    //![vina_gaussian2]
}

TEST_CASE("vina computes repulsion", "[vina_repulsion]") {
    Vina vina;
    Real e;
    Real f;

    e = vina.vina_repulsion(-1, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(-2, 1e-4));

    e = vina.vina_repulsion(0, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));

    e = vina.vina_repulsion(1, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));

    //![vina_repulsion] 
}


TEST_CASE("vina computes hydrophobic", "[vina_hydrophobic]") {
    Vina vina;
    Real e;
    Real f;

    e = vina.vina_hydrophobic(0.2, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));

    e = vina.vina_hydrophobic(1, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0.5, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(-1, 1e-4));

    e = vina.vina_hydrophobic(2, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));
    //![vina_hydrophobic]
}

TEST_CASE("vina computes hbond", "[vina_hbond]") {
    Vina vina;
    Real e;
    Real f;

    e = vina.vina_hbond(-1, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));

    e = vina.vina_hbond(-0.5, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0.714285714286, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(-1.42857142857, 1e-4));

    e = vina.vina_hbond(0.1, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));
    //![vina_hbond]
}


TEST_CASE("vina computes vina_score", "[eval_ef]") {
    Vina vina;
    Real e;
    Real f;
    Real d = 0;
    e = vina.eval_ef(d, VN_TYPE_C_P, VN_TYPE_C_P, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(-0.036122, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(-0.000815, 1e-4));
    // for (int i = 0; i < 9; i++){
    //     Real delta = std::pow(10, -i);
    //     printf("delta=%.10f, f_DF=%.10f\n", delta,
    //         cal_f_by_fd(std::bind(&Vina::eval_ef, &vina, std::placeholders::_1, VN_TYPE_C_P, VN_TYPE_C_P, std::placeholders::_2), d, delta)
    //     );
    // }

    //![eval_ef]
}




