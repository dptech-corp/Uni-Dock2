//
// Created by Congcong Liu on 24-9-23.
//

#include "spdlog/spdlog.h"
#include <string>
#include <set>
#include <fstream>
#include <rapidjson/document.h>
#include "rapidjson/filewritestream.h"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <rapidjson/prettywriter.h>

#include "json.h"
#include "rapidjson_parser.h"
#include "model/model.h"
#include "myutils/mymath.h"
#include "myutils/common.h"

#include <numeric>
#include <algorithm>

#include "constants/constants.h"

namespace rj = rapidjson;

rj::Document parse_json(const std::string& fp){
    rj::Document doc;
    std::ifstream ifs(fp);
    std::string json_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    doc.Parse(json_str.c_str());

    if (doc.HasParseError()){
        throw std::runtime_error("rapidjson: Failed to parse JSON file: " + fp);
    }

    return doc;
}



void read_ud_from_json(const std::string& fp, const Box& box, UDFixMol& out_fix, UDFlexMolList& out_flex_list,
                       std::vector<std::string>& out_fns_flex, bool use_tor_lib) {
    rj::Document doc = parse_json(fp);
    spdlog::info("Json is successfully parsed");

    //---------------- Parse score types ----------------
    const auto& scores = doc["score"].GetArray();
    std::set<std::string> score_types;
    for (const auto& score : scores) {
        score_types.insert(score.GetString());
    }

    // Use RapidJsonParser to parse the data
    RapidJsonParser parser(doc);
    
    // Parse receptor
    parser.parse_receptor_info(box, out_fix);
    
    // Parse ligands
    parser.parse_ligands_info(out_flex_list, out_fns_flex, use_tor_lib);
    
    // Add inter pairs for each ligand (this needs to be done after receptor is parsed)
    for (auto& flex_mol : out_flex_list) {
        // inter pairs: flex v.s. receptor
        for (int i = 0; i < flex_mol.natom; i++) {
            if (flex_mol.vina_types[i] == VN_TYPE_H) { // ignore Hydrogen on ligand and protein
                continue;
            }
            for (int j = 0; j < out_fix.natom; j++) {
                if (out_fix.vina_types[j] == VN_TYPE_H) {
                    continue;
                }
                flex_mol.inter_pairs.push_back(i);
                flex_mol.inter_pairs.push_back(j);
            }
        }
    }
    
    spdlog::debug("Json is Done.");
}

void write_poses_to_json(std::string fp_json, const std::vector<std::string>& flex_names,
                         const std::vector<std::vector<int>>& filtered_pose_inds_list,
                         const FlexPose* flex_pose_list_res,
                         const Real* flex_pose_list_real_res,
                         const std::vector<int>& list_i_real){
    rj::Document doc;
    doc.SetObject();

    // add poses

    for (int i = 0; i < flex_names.size(); i++){
        auto flex_name = flex_names[i];
        rj::Value flex_data(rj::kArrayType);

        for (auto& j: filtered_pose_inds_list[i]){
            rj::Value pose_obj;
            pose_obj.SetObject();

            rj::Value energy(rj::kArrayType);
            energy.PushBack(flex_pose_list_res[j].rot_vec[0], doc.GetAllocator()); // affinity
            energy.PushBack(flex_pose_list_res[j].rot_vec[1], doc.GetAllocator()); // total = intra + inter
            energy.PushBack(flex_pose_list_res[j].center[0], doc.GetAllocator()); // intra
            energy.PushBack(flex_pose_list_res[j].center[1], doc.GetAllocator()); // inter
            energy.PushBack(flex_pose_list_res[j].center[2], doc.GetAllocator()); // penalty
            energy.PushBack(flex_pose_list_res[j].rot_vec[3], doc.GetAllocator()); // conf independent contribution
            pose_obj.AddMember("energy", energy.Move(), doc.GetAllocator());

            rj::Value coords(rj::kArrayType);
            for (int k = list_i_real[j * 2]; k < list_i_real[j * 2 + 1]; k++){
                coords.PushBack(flex_pose_list_real_res[k], doc.GetAllocator());
            }
            pose_obj.AddMember("coords", coords.Move(), doc.GetAllocator());

            rj::Value dihedrals(rj::kArrayType);
            for (int k = list_i_real[j * 2 + 1]; k < list_i_real[j * 2 + 2]; k++){
                dihedrals.PushBack(rad_to_ang(flex_pose_list_real_res[k]), doc.GetAllocator());
            }
            pose_obj.AddMember("dihedrals", dihedrals.Move(), doc.GetAllocator());

            flex_data.PushBack(pose_obj.Move(), doc.GetAllocator());
        }

        rj::Value key(flex_name.c_str(), doc.GetAllocator());
        doc.AddMember(key, flex_data, doc.GetAllocator());
    }

    // write to file
    char writeBuffer[65536];
    FILE* f = fopen(fp_json.c_str(), "w");
    rj::FileWriteStream os(f, writeBuffer, sizeof(writeBuffer));
    rj::Writer<rj::FileWriteStream> writer(os);
    writer.SetMaxDecimalPlaces(3);
    doc.Accept(writer);

    fclose(f);
}
