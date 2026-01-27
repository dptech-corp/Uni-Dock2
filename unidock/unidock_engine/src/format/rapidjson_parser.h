#ifndef FORMAT_RJSON_PARSER_H
#define FORMAT_RJSON_PARSER_H

#include "model/model.h"
#include <rapidjson/document.h>

class RapidJsonParser {
public:
    RapidJsonParser(const rapidjson::Document& doc): doc_(doc){};
    
    void parse_receptor_info(const Box& box_protein, UDFixMol& fix_mol);
    void parse_ligands_info(UDFlexMolList& flex_mol_list, std::vector<std::string>& fns_flex, bool use_tor_lib);

private:
    const rapidjson::Document& doc_;
    
    void split_torsions_into_frags(const std::set<int>& root, const std::vector<UDTorsion>& torsions,
                                   std::vector<std::set<int>>& out_frags);
};



#endif // FORMAT_RJSON_PARSER_H