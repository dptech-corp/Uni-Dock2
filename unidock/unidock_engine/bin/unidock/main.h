//
// Created by Congcong Liu on 25-6-10.
//

#ifndef MAIN_H
#define MAIN_H

#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include "screening/core.h"

// Macro to emit a parameter with its default value and documentation
#define EMIT_PARAM_DEFAULT(emitter, name) \
    emitter << YAML::Key << #name \
            << YAML::Value << CoreInputDefaults::name \
            << YAML::Comment(CoreInputDocs::name)

#define EMIT_PARAM(emitter, name, value, comment) \
    emitter << YAML::Key << #name \
            << YAML::Value << value \
            << YAML::Comment(comment)

#define EMIT_PARAM_REQUIRED(emitter, name, comment) \
    emitter << YAML::Key << #name \
            << YAML::Comment(comment)

/**
 * @brief Generate and dump the default config template using YAML::Emitter.
 * Documentation and default values come from CoreInputDefaults and CoreInputDocs.
 */
inline void dump_config_template(const std::string& path) {
    YAML::Emitter out;
    out.SetFloatPrecision(3);  // decimal places
    out.SetDoublePrecision(3); 
    out << YAML::BeginMap;

    // ==================== Settings ====================
    out << YAML::Key << "Settings";
    out << YAML::Value << YAML::BeginMap;

    EMIT_PARAM_DEFAULT(out, task);
    EMIT_PARAM_DEFAULT(out, search_mode);
    EMIT_PARAM_DEFAULT(out, constraint_docking);

    // Box parameters (required, use placeholder values)
    EMIT_PARAM(out, center_x, 0.0, "float: X coordinate of box center (Angstrom) - REQUIRED");
    EMIT_PARAM(out, center_y, 0.0, "float: Y coordinate of box center (Angstrom) - REQUIRED");
    EMIT_PARAM(out, center_z, 0.0, "float: Z coordinate of box center (Angstrom) - REQUIRED");
    EMIT_PARAM(out, size_x, 30.0, "float: Box size along X axis (Angstrom) - REQUIRED");
    EMIT_PARAM(out, size_y, 30.0, "float: Box size along Y axis (Angstrom) - REQUIRED");
    EMIT_PARAM(out, size_z, 30.0, "float: Box size along Z axis (Angstrom) - REQUIRED");

    out << YAML::EndMap;  // Settings

    // ==================== Advanced ====================
    out << YAML::Key << "Advanced";
    out << YAML::Value << YAML::BeginMap;

    EMIT_PARAM_DEFAULT(out, seed);
    EMIT_PARAM_DEFAULT(out, exhaustiveness);
    EMIT_PARAM_DEFAULT(out, randomize);
    EMIT_PARAM_DEFAULT(out, mc_steps);
    EMIT_PARAM_DEFAULT(out, opt_steps);
    EMIT_PARAM_DEFAULT(out, refine_steps);
    EMIT_PARAM_DEFAULT(out, rmsd_limit);
    EMIT_PARAM_DEFAULT(out, num_pose);
    EMIT_PARAM_DEFAULT(out, energy_range);
    EMIT_PARAM_DEFAULT(out, bias);
    EMIT_PARAM_DEFAULT(out, bias_k);
    EMIT_PARAM_DEFAULT(out, use_tor_lib);
    out << YAML::EndMap;  // Advanced

    // ==================== Hardware ====================
    out << YAML::Key << "Hardware";
    out << YAML::Value << YAML::BeginMap;

    EMIT_PARAM_DEFAULT(out, gpu_device_id);
    EMIT_PARAM_DEFAULT(out, max_gpu_memory);

    out << YAML::EndMap;  // Hardware

    // ==================== Outputs ====================
    out << YAML::Key << "Outputs";
    out << YAML::Value << YAML::BeginMap;

    EMIT_PARAM_DEFAULT(out, output_dir);

    out << YAML::EndMap;  // Outputs

    // ==================== Inputs ====================
    out << YAML::Key << "Inputs";
    out << YAML::Value << YAML::BeginMap;

    EMIT_PARAM_REQUIRED(out, json, "str: Input JSON file - REQUIRED");

    out << YAML::EndMap;  // Inputs

    // ====================
    out << YAML::EndMap;  // root

    // Write to file
    std::ofstream fout(path);
    if (!fout) {
        std::cerr << "Failed to create config template file: " << path << std::endl;
        exit(1);
    }
    fout << "# Uni-Dock2 Configuration Template\n";
    fout << "# Generated with default values from CoreInputDefaults\n\n";
    fout << out.c_str();
}

#undef EMIT_PARAM_DEFAULT
#undef EMIT_PARAM
#undef EMIT_PARAM_REQUIRED
#endif //MAIN_H
