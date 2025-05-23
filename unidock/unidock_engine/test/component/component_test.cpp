//
// Created by Congcong Liu on 24-10-15.
//

#include <catch2/catch_amalgamated.hpp>
#include "myutils/errors.h"


int main(int argc, char* argv[]) {
    // Change the logging level as needed
    init_logger("component_test.log", 1);

    int result = Catch::Session().run(argc, argv);
    return result;
}
