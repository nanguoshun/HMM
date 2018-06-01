//
// Created by  ngs on 27/04/2018.
//

#ifndef CLIONTEST_DATASETINFO_H
#define CLIONTEST_DATASETINFO_H

#include <vector>
#include <set>
#include <map>
#include <fstream>
#include <iostream>
#include <string>
#include "common.h"
class Data{

};
class DatasetMgr{

    public:
    explicit DatasetMgr(bool is_sentence_level);
    ~DatasetMgr();
    bool OpenDataSet(const char *file_name, bool is_training);

    void OpenTrainSet(std::vector<std::string> *ptr_vector, bool is_sentence_level);
    void OpenTestSet(std::vector<std::string> *ptr_vector);
    bool Tokenized(char* ptr_line, const char* ptr_space, std::vector<std::string> *ptr_string_line, size_t tag_maxsize, bool istraining);
    void GenerateStateTransitionVector();
    bool MergeTwoString(std::string *ptr_str1, std::string str2, std::string separator);
    void GenerateCountMap(std::vector<std::string> *ptr_vector, std::map<std::string, size_t> *ptr_count_map, bool option);
    void CalcProb(std::map<std::string, size_t> *ptr_cout, std::map<std::string, size_t> *ptr_trans_cout, std::map<std::string, double > *ptr_prob);
    void Calc();
    std::map<std::string, double> *GetStateTransProbMap() const;
    std::map<std::string, double> *GetEmissionProbMap() const;
    std::vector<std::string> *GetTestFlagVector() const;
    std::vector<std::string> *GetTestFeatureVector() const;
    std::set<std::string> *GetTagSet() const;
    std::vector<std::string> *GetTrainingXVector() const;
    std::set<std::string> *GetTrainingXSet() const;
    size_t GetNumOfTrainingSeqs() const;


private:
    char *ptr_line_;
    std::set<std::string> *ptr_tag_set_;
    //tag of training dateset
    std::vector<std::string> *ptr_tag_vector_;
    std::vector<std::string> *ptr_pair_tag_vector_;
    std::map<std::string, size_t> *ptr_tag_count_map_;
    std::map<std::string, size_t> *ptr_pair_tag_count_map_;
    std::map<std::string, double > *ptr_state_to_state_prob_map_;
    //feature of training dateset
    std::vector<std::string> *ptr_x_tag_vector_;
    std::map<std::string, size_t> *ptr_x_tag_count_map_;
    std::map<std::string, double > *ptr_state_to_x_prob_map_;
    std::vector<std::string> *ptr_x_vector_;
    std::set<std::string> *ptr_x_set_;

    size_t  num_of_training_setence_;
    //chunk level POS or sentence level POS.
    bool is_sentence_level_;
    //test dataset
    std::vector<std::string> *ptr_test_x_vector_;
    std::vector<std::string> *ptr_test_tag_vector_;


};

#endif //CLIONTEST_DATASETINFO_H


