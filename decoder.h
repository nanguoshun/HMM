//
// Created by  ngs on 28/04/2018.
//

#ifndef CLIONTEST_DECODER_H
#define CLIONTEST_DECODER_H

#include <iostream>
#include "datasetmgr.h"
struct Node{
    std::string state;
    std::string pre_state;
    std::string path;
    double prob;
};

class Decoder{
public:
    explicit Decoder(const char* test_file_name, DatasetMgr *ptr_datasetmgr);
    ~Decoder();
    void Decoding();
    void Init();
    std::string& SearchStateOfMaxProb(std::string str_x);
    void Vertibi(std::vector<std::string> obsersvation_str);
    double FindProb(std::map<std::string, double > &prob_map, std::string key_str);
    double GetEmissionProb(std::map<std::string, double > &prob_map,std::string observation_str, std::string state_str);
    void GetNode(std::string state, Node &node);
    void ResetNode(std::string state, Node &node);
    void Output(std::vector<std::string> observation_sentence);
private:
    DatasetMgr *ptr_datasetmgr_;
    std::map<std::string, double > *ptr_state_to_state_prob_map_;
    std::map<std::string, double > *ptr_state_to_x_prob_map_;
    std::vector<std::string> *ptr_test_x_vector_;
    std::vector<std::string> *ptr_test_tag_vector_;
    std::set<std::string> *ptr_tag_set_;

    std::vector<std::string> *ptr_path_str_vector_;
    std::vector<std::string> *ptr_path_prob_vector_;

    std::map<std::string, std::string> *ptr_x_to_state_docoding_;
    std::map<std::string, Node> *ptr_state_path_;
};

#endif //CLIONTEST_DECODER_H
