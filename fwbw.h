//
// Created by  ngs on 01/06/2018.
//

#ifndef HMM_FWBW_H
#define HMM_FWBW_H
#include "common.h"
#include <map>

class FB{
public:
    FB();
    double Forward(std::map<std::string, int> *ptr_x_corpus_map, std::vector<std::string> seq, std::pair<int, int> u_j, HMMParameters parameters);
    double BackWard(std::map<std::string, int> *ptr_x_corpus_map, std::vector<std::string> seq, std::pair<int, int> u_j, HMMParameters parameters);
};

#endif //HMM_FWBW_H
