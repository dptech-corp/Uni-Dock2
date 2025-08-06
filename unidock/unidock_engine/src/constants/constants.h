//
// Created by Congcong Liu on 24-9-26.
//

#ifndef CONSTANT_H
#define CONSTANT_H

#define LINE_SEARCH_STEPS 10
#define LINE_SEARCH_C0 0.0001
#define LINE_SEARCH_MULTIPLIER 0.5


#define PENALTY_SLOPE 1e6

const Real VN_VDW_RADII[] = {
    1.1,  // H <from 10.1021/jp8111556>
    1.92, // B <from 10.1021/jp8111556>
    1.9,  // C_H // the following from AutoDock Vina
    1.9,  // C_P
    1.8,  // N_P
    1.8,  // N_D
    1.8,  // N_A
    1.8,  // N_DA
    1.7,  // O_P
    1.7,  // O_D
    1.7,  // O_A
    1.7,  // O_DA
    1.5,  // F_H
    2.2,  // Si
    2.1,  // P_P
    2.0,  // S_P
    1.8,  // Cl_H
    2.0,  // Br_H
    2.2,  // I_H
    2.3,  // At
    1.2,  // Met_D
};



// Vina Type (XS-score in AutoDock Vina)
constexpr int VN_TYPE_H = 0;
constexpr int VN_TYPE_B = 1;
constexpr int VN_TYPE_C_H = 2;
constexpr int VN_TYPE_C_P = 3;
constexpr int VN_TYPE_N_P = 4;
constexpr int VN_TYPE_N_D = 5;
constexpr int VN_TYPE_N_A = 6;
constexpr int VN_TYPE_N_DA = 7;
constexpr int VN_TYPE_O_P = 8;
constexpr int VN_TYPE_O_D = 9;
constexpr int VN_TYPE_O_A = 10;
constexpr int VN_TYPE_O_DA = 11;
constexpr int VN_TYPE_F_H = 12;
constexpr int VN_TYPE_Si = 13;  // Silicon
constexpr int VN_TYPE_P_P = 14;
constexpr int VN_TYPE_S_P = 15;
constexpr int VN_TYPE_Cl_H = 16;
constexpr int VN_TYPE_Br_H = 17;
constexpr int VN_TYPE_I_H = 18;
constexpr int VN_TYPE_At = 19;  // Astatine
constexpr int VN_TYPE_Met_D = 20; // metals and other default elements



#endif //CONSTANT_H




