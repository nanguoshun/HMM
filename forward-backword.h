//
// Created by  ngs on 12/05/2018.
//

#ifndef HMM_FORWARD_H
#define HMM_FORWARD_H

#include <vector>
#include <map>
#include "common.h"

class ForwardBackward {
public:
    explicit ForwardBackward();

    double ForwardResult(std::map<std::string, int> *ptr_x_corpus_map, std::vector<std::string> seq, std::pair<int, int> u_j, HMMParameters parameters);
    double BackwardResult(std::map<std::string, int> *ptr_x_corpus_map,std::vector<std::string> seq, std::pair<int, int> u_j, HMMParameters parameters);
};

#endif //HMM_FORWARD_H
