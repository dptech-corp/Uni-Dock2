#ifndef SCREENING_H
#define SCREENING_H

#include <array>
#include <iostream>
#include "model/model.h"


// ----------------------------------------------  Config ----------------------------------------------
struct Config {
    int MAX_NUM_OF_ATOMS;
    int MAX_NUM_OF_TORSION;

    constexpr Config(int num_atom, int num_torsion):
        MAX_NUM_OF_ATOMS(num_atom),
        MAX_NUM_OF_TORSION(num_torsion) {}
};

static constexpr Config SmallConfig{40, 8};
static constexpr Config MediumConfig{80, 16};
static constexpr Config LargeConfig{120, 24};
static constexpr Config ExtraLargeConfig{160, 32};

static constexpr int NGroup = 4;
static constexpr std::array<const char*, NGroup + 1> GroupNames = {"small", "medium", "large", "extra", "overflow"};
static constexpr std::array<Config, NGroup> Configs = {SmallConfig, MediumConfig, LargeConfig, ExtraLargeConfig};
static constexpr int NatomThresholds[NGroup] = {SmallConfig.MAX_NUM_OF_ATOMS, MediumConfig.MAX_NUM_OF_ATOMS,
    LargeConfig.MAX_NUM_OF_ATOMS, ExtraLargeConfig.MAX_NUM_OF_ATOMS};
static constexpr int NTorsionThresholds[NGroup] = {SmallConfig.MAX_NUM_OF_TORSION, MediumConfig.MAX_NUM_OF_TORSION,
    LargeConfig.MAX_NUM_OF_TORSION, ExtraLargeConfig.MAX_NUM_OF_TORSION};




void run_screening(UDFixMol & dpfix_mol, UDFlexMolList &dpflex_mols, const std::vector<std::string>& fns_flex,
                   const std::string &dp_out, DockParam& dock_param, int device_max_memory, std::string name_json = "ud2");





#endif
