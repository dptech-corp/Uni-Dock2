#ifndef FORMAT_RJSON_PARSER_H
#define FORMAT_RJSON_PARSER_H

#include "parser_base.h"
#include <rapidjson/document.h>

class RapidJsonParser : public ParserBase {
public:
    RapidJsonParser(const rapidjson::Document& doc);
    
    void parse_receptor_info(const Box& box_protein, UDFixMol& fix_mol) override;
    void parse_ligands_info(UDFlexMolList& flex_mol_list, std::vector<std::string>& fns_flex, bool use_tor_lib) override;

private:
    const rapidjson::Document& doc_;
    
    void split_torsions_into_frags(const std::set<int>& root, const std::vector<UDTorsion>& torsions,
                                   std::vector<std::set<int>>& out_frags);
};
#endif // FORMAT_RJSON_PARSER_H