//
// Created by Congcong Liu on 24-11-27.
//

#include <catch2/catch_amalgamated.hpp>
#include "format/json.h"
#include "myutils/mymath.h"

TEST_CASE("Test JSON", "[json]"){
    std::string json_path = std::string(TEST_DATA_DIR) + "/dev1/simple.json";
    UDFixMol fix_mol;
    UDFlexMolList flex_mol_list;
    std::vector<std::string> flex_fns;
    Real l = 100;
    Box box{-l,l, -l, l, -l, l};

    read_ud_from_json(json_path, box, fix_mol, flex_mol_list, flex_fns);

    // check all members of fix_mol
    REQUIRE(fix_mol.natom == 5);
    REQUIRE(fix_mol.coords.size() == 5 * 3);
    REQUIRE(fix_mol.vina_types.size() == 5);
    REQUIRE(fix_mol.ff_types.size() == 5);
    REQUIRE(fix_mol.charges.size() == 5);

    // check all members of flex_mol_list
    REQUIRE(flex_mol_list.size() == 2);
    REQUIRE(flex_fns.size() == 2);
    REQUIRE(flex_fns[0] == "lig1");
    REQUIRE(flex_fns[1] == "lig2");

    // check all members of flex_mol
    auto& flex_mol = flex_mol_list[0];
    int natom = 3;
    REQUIRE(flex_mol.natom == natom);
    REQUIRE(flex_mol.coords.size() == natom * 3);
    REQUIRE_THAT(flex_mol.center[0], Catch::Matchers::WithinAbs(65.484336, 1e-4));
    REQUIRE_THAT(flex_mol.center[1], Catch::Matchers::WithinAbs(14.577332, 1e-4));
    REQUIRE_THAT(flex_mol.center[2], Catch::Matchers::WithinAbs(43.487335, 1e-4));
    REQUIRE(flex_mol.vina_types.size() == natom);
    REQUIRE(flex_mol.ff_types.size() == natom);
    REQUIRE(flex_mol.charges.size() == natom);
    REQUIRE(flex_mol.dihedrals.size() == 2);
    REQUIRE(flex_mol.dihedrals[0] == ang_to_rad(177));
    REQUIRE(flex_mol.dihedrals[1] == ang_to_rad(70));
    REQUIRE(flex_mol.torsions.size() == 2);
    REQUIRE(flex_mol.pairs_1213.size() == 8);
    REQUIRE(flex_mol.pairs_14.size() == 9);
    REQUIRE(flex_mol.intra_pairs.size() == 1 * 2);

    flex_mol = flex_mol_list[1];
    natom = 2;
    REQUIRE(flex_mol.natom == natom);
    REQUIRE(flex_mol.coords.size() == natom * 3);
    REQUIRE_THAT(flex_mol.center[0], Catch::Matchers::WithinAbs(63.788497, 1e-4));
    REQUIRE_THAT(flex_mol.center[1], Catch::Matchers::WithinAbs(12.872499, 1e-4));
    REQUIRE_THAT(flex_mol.center[2], Catch::Matchers::WithinAbs(41.792999, 1e-4));
    REQUIRE(flex_mol.vina_types.size() == natom);
    REQUIRE(flex_mol.ff_types.size() == natom);
    REQUIRE(flex_mol.charges.size() == natom);
    REQUIRE(flex_mol.dihedrals.size() == 2);
    REQUIRE(flex_mol.dihedrals[0] == ang_to_rad(-13.5));
    REQUIRE(flex_mol.dihedrals[1] == ang_to_rad(110));
    REQUIRE(flex_mol.torsions.size() == 2);
    REQUIRE(flex_mol.pairs_1213.size() == 5);
    REQUIRE(flex_mol.pairs_14.size() == 6);
    REQUIRE(flex_mol.intra_pairs.size() == 1 * 2);
    //![json]
}


