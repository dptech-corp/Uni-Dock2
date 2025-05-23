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

SCOPE_INLINE std::pair<int, int> order_pair(int a, int b){
    return std::make_pair(std::min(a, b), std::max(a, b));
}

bool checkInAnySameSet(const std::vector<std::set<int>>& frags, int v1, int v2) {
    for (auto frag: frags) {
        if (frag.find(v1) != frag.end() and frag.find(v2) != frag.end()) {
            return true;
        }
    }
    return false;
}


/**
 * A temporary patch to help exclude intra pairs where two atoms belong to the same fragment.
 * @param root Root atoms.
 * @param torsions
 * @param out_frags
 */
void split_torsions_into_frags(const std::set<int>& root, const std::vector<UDTorsion>& torsions,
                               std::vector<std::set<int>>& out_frags) {
    // The first frag is the root atoms
    std::set<int> frag0;
    for (int i : root){
        frag0.insert(i);
    }

    // use index list
    std::vector<int> range(torsions.size());
    std::iota(range.begin(), range.end(), 0);

    // Step 1: Identify big torsions and add them as fragments
    std::vector<int> range_tmp = range;
    for (int itor : range_tmp) {
        auto& torsion = torsions[itor];
        // Check if either axis atom is in the root set
        if (root.find(torsion.axis[0]) != root.end() || root.find(torsion.axis[1]) != root.end()) {
            // remove this itor from the list
            range.erase(std::remove(range.begin(), range.end(), itor), range.end());

            // also add axis to the root frag
            frag0.insert(torsion.axis[0]);
            frag0.insert(torsion.axis[1]);

            // Create a new fragment for this big torsion
            std::set<int> frag_tmp;
            // add to this new frag
            for (int i : torsion.rotated_atoms){
                frag_tmp.insert(i);
            }
            // also add axis to this frag
            frag_tmp.insert(torsion.axis[0]);
            frag_tmp.insert(torsion.axis[1]);
            out_frags.push_back(frag_tmp);
        }
    }
    out_frags.push_back(frag0);


    // Step 2: Split existing fragments using other torsions
    for (int itor : range) {
        auto& torsion = torsions[itor];

        // Check if the torsion's axis atoms belong to an existing fragment
        for (auto& frag : out_frags) {
            if ((frag.find(torsion.axis[0]) != frag.end()) and (frag.find(torsion.axis[1]) != frag.end())) {
                // Split the fragment into two parts
                std::set<int> frag1, frag2;
                std::set<int> tor_set(torsion.rotated_atoms.begin(), torsion.rotated_atoms.end());
                std::set_intersection(frag.begin(), frag.end(),
                                      tor_set.begin(), tor_set.end(),
                                      std::inserter(frag1, frag1.begin()));
                std::set_difference(frag.begin(), frag.end(),
                                    tor_set.begin(), tor_set.end(),
                                    std::inserter(frag2, frag2.begin()));

                // also add the axis
                frag1.insert(torsion.axis[0]);
                frag1.insert(torsion.axis[1]);
                frag2.insert(torsion.axis[0]);
                frag2.insert(torsion.axis[1]);

                // Replace the original fragment with the two new fragments
                frag = frag1;
                out_frags.push_back(frag2);
                break; // Move to the next torsion
            }
        }
    }
}


void read_ud_from_json(const std::string& fp, const Box& box, UDFixMol& out_fix, UDFlexMolList& out_flex_list,
                       std::vector<std::string>& out_fns_flex, bool use_tor_lib){
    rj::Document doc = parse_json(fp);
    spdlog::info("Json is successfully parsed");
    Real x, y, z;
    int type;

    //---------------- Parse score types ----------------
    const auto& scores = doc["score"].GetArray();
    std::set<std::string> score_types;
    for (const auto& score : scores){
        score_types.insert(score.GetString());
    }

    //---------------- Parse receptor ----------------  
    const auto& receptor = doc["receptor"].GetArray();
    for (const auto& atom_info : receptor){
        type = atom_info[3].GetInt();
        if (type == VN_TYPE_H){ // remove Hydrogens
            continue;
        }

        x = atom_info[0].GetFloat();
        y = atom_info[1].GetFloat();
        z = atom_info[2].GetFloat();
        if (box.is_inside(x, y, z)){ // for acceleration
            out_fix.coords.push_back(x);
            out_fix.coords.push_back(y);
            out_fix.coords.push_back(z);
            out_fix.vina_types.push_back(type);
            out_fix.ff_types.push_back(atom_info[4].GetInt());
            out_fix.charges.push_back(atom_info[5].GetFloat());
        }
    }
    out_fix.natom = out_fix.charges.size();
    spdlog::debug("Json data: Receptor is successfully extracted.");


    //---------------- Parse ligands and file names ----------------
    // other keys are deemed as flexible molecule names
    std::set<std::string> exclude_keys = {"score", "receptor"};
    for (auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it){
        std::string key = it->name.GetString();

        if (exclude_keys.find(key) == exclude_keys.end()){
            // This is a flexible molecule
            out_fns_flex.push_back(key);

            UDFlexMol flex_mol;
            const auto& v = it->value.GetObject();
            Real coords_sum[3] = {0};
            // Atoms
            // spdlog::debug("Json data: extract Flex-Atom...");
            const auto& atom_info = v["atoms"].GetArray();
            for (int ia = 0; ia < atom_info.Size(); ia++){
                const auto& atom_line = atom_info[ia].GetArray();
                x = atom_line[0].GetFloat();
                y = atom_line[1].GetFloat();
                z = atom_line[2].GetFloat();
                flex_mol.coords.push_back(x);
                flex_mol.coords.push_back(y);
                flex_mol.coords.push_back(z);
                coords_sum[0] += x;
                coords_sum[1] += y;
                coords_sum[2] += z;

                flex_mol.vina_types.push_back(atom_line[3].GetInt());
                flex_mol.ff_types.push_back(atom_line[4].GetInt());
                flex_mol.charges.push_back(atom_line[5].GetFloat());

                // dev: check 1-2&1-3 prior to 1-4; no self atom; no duplicated atom.
                for (const auto& a : atom_line[6].GetArray()){
                    auto ib = a.GetInt();
                    if (ia == ib){
                        spdlog::warn("Remove wrong pair 1-2 & 1-3 for Self: flex {} atom {} - {}", key, ia, ib);
                    }
                    else{
                        flex_mol.pairs_1213.insert(order_pair(ia, ib));
                    }
                }
                for (const auto& a : atom_line[7].GetArray()){
                    auto ib = a.GetInt();
                    if (ia == ib){
                        spdlog::warn("Remove wrong pair 1-4 for Self: flex {} atom {} - {}", key, ia, ib);
                    }
                    else{
                        auto p = order_pair(ia, ib);
                        if (flex_mol.pairs_1213.find(p) != flex_mol.pairs_1213.end()){
                            spdlog::warn("Remove wrong pair 1-4 for Already in 1-2&1-3: flex {} atom {} - {}", key, ia,
                                         ib);
                        }
                        else{
                            flex_mol.pairs_14.insert(p);
                        }
                    }
                }
            }

            std::set<int> root_atoms;
            const auto& root_info = v["root_atoms"].GetArray();
            for (const auto& a : root_info){
                root_atoms.insert(a.GetInt());
            }

            // Torsions
            // spdlog::debug("Json data: extract Flex-Torsion...");
            const auto& torsions_info = v["torsions"].GetArray();
            for (const auto& torsion_info : torsions_info){
                const auto& tor_info = torsion_info.GetArray();
                UDTorsion torsion;

                // spdlog::debug("    extract torsion atoms...");
                for (int i = 0; i < 4; i++){
                    // four atoms
                    torsion.atoms[i] = tor_info[0].GetArray()[i].GetInt();
                }

                // axis is the two middle atoms
                torsion.axis[0] = torsion.atoms[1];
                torsion.axis[1] = torsion.atoms[2];

                // dihedral value
                // spdlog::debug("    extract dihedral...");
                flex_mol.dihedrals.push_back(ang_to_rad(tor_info[1].GetFloat()));

                // range list
                // spdlog::debug("    extract range list...");
                if (use_tor_lib){
                    for (const auto& r : tor_info[2].GetArray()){
                        Real rad_lo = ang_to_rad(r.GetArray()[0].GetFloat());
                        Real rad_hi = ang_to_rad(r.GetArray()[1].GetFloat());
                        // For range crossing "180/-180", split it into to ranges
                        if (rad_lo < rad_hi){
                            torsion.range_list.push_back(rad_lo);
                            torsion.range_list.push_back(rad_hi);
                        } else if (rad_lo > 0 && rad_hi < 0){
                            torsion.range_list.push_back(rad_lo);
                            torsion.range_list.push_back(PI);
                            torsion.range_list.push_back(-PI);
                            torsion.range_list.push_back(rad_hi);
                        } else{
                            spdlog::critical("Input json has wrong range list, rad_lo > rad_hi: {}, {}",
                                r.GetArray()[0].GetFloat(), r.GetArray()[1].GetFloat());
                        }
                    }
                } else{ // TODO: consider moving this switch outside of C++ engine?
                    torsion.range_list.push_back(-PI);
                    torsion.range_list.push_back(PI);
                }

                // rotated atoms
                // spdlog::debug("    extract rotated atoms...");
                for (const auto& a : tor_info[3].GetArray()){
                    torsion.rotated_atoms.push_back(a.GetInt());
                }
                // gaff2 parameters, may be multiple groups
                // spdlog::debug("    extract gaff2 parameters...");
                for (const auto& gaff2 : tor_info[4].GetArray()){
                    torsion.param_gaff2.push_back(gaff2.GetArray()[0].GetFloat());
                    torsion.param_gaff2.push_back(gaff2.GetArray()[1].GetFloat());
                    torsion.param_gaff2.push_back(gaff2.GetArray()[2].GetFloat());
                    torsion.param_gaff2.push_back(gaff2.GetArray()[3].GetFloat());
                }
                // add this torsion to the flex_mol
                flex_mol.torsions.push_back(torsion);
            }

            std::vector<std::set<int>> frags;
            split_torsions_into_frags(root_atoms, flex_mol.torsions, frags);
            // Compute necessary properties
            // spdlog::debug("Json data: compute properties...");
            flex_mol.name = key;
            flex_mol.natom = flex_mol.charges.size();
            flex_mol.center[0] = coords_sum[0] / flex_mol.natom;
            flex_mol.center[1] = coords_sum[1] / flex_mol.natom;
            flex_mol.center[2] = coords_sum[2] / flex_mol.natom;

            // intra pairs
            for (int i = 0; i < flex_mol.natom; i++){
                for (int j = i + 1; j < flex_mol.natom; j++){
                    // exclude 1-2, 1-3, 1-4 pairs
                    if ((flex_mol.pairs_1213.find(order_pair(i, j)) == flex_mol.pairs_1213.end()) and
                        (flex_mol.pairs_14.find(order_pair(i, j)) == flex_mol.pairs_14.end()) and
                        (!checkInAnySameSet(frags, i, j)) //
                        ){
                        flex_mol.intra_pairs.push_back(i);
                        flex_mol.intra_pairs.push_back(j);
                    }
                }
            }
            // inter pairs: flex v.s. receptor
            for (int i = 0; i < flex_mol.natom; i++){
                if (flex_mol.vina_types[i] == VN_TYPE_H){ //ignore Hydrogen on ligand and protein
                    continue;
                }
                for (int j = 0; j < out_fix.natom; j++){
                    if (out_fix.vina_types[j] == VN_TYPE_H){
                        continue;
                    }
                    flex_mol.inter_pairs.push_back(i);
                    flex_mol.inter_pairs.push_back(j);
                }
            }

            // add this flex_mol to the list
            out_flex_list.push_back(flex_mol);
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
