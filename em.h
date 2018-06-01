//
// Created by  ngs on 12/05/2018.
//

#ifndef HMM_EM_H
#define HMM_EM_H

#include "forward-backword.h"
#include <vector>
#include "datasetmgr.h"
#include <cmath>

class EM{

public:
    explicit EM(const char* file_name, DatasetMgr *ptr_datamgr);
    ~EM();
    void Init();
    void EStep();
    void MStep();
    double CalcCountOfUV(std::pair<int,int> uv);
    double CalcPX(std::vector<std::string> seq);
    double CalcCountOfU(int u);
    double CalcCountOfUK(std::pair<int, int> uk);
    void GenerateSeqFromVector(std::vector<std::string> *ptr_vector, std::vector<std::vector<std::string>> *ptr_seq_vector);
    void UpdateParameters();
    void Learning();
    bool IsIteration();
    void RandomInitProb(double *ptr_probarray, int array_size);
private:
    DatasetMgr *ptr_datamgr_;
    std::set<std::string> *ptr_x_set_;
    std::vector<std::string> *ptr_train_x_vector_; //sequence is separated by a FLAG;
    std::vector<std::vector<std::string>> *ptr_training_seq_;

    ForwardBackward *ptr_fwbw;
    size_t  number_of_x_;
    size_t  num_of_training_setence_;
    HMMParameters HMM_Parameters_;
    //std::vector<std::vector<double>> **pptr_alpha_;
    //std::vector<std::vector<double>> **pptr_beta_;
    std::map<std::string, int> *ptr_x_corpus_map_;
    std::vector<std::string> *ptr_x_corpus_;

    std::vector<double > *ptr_Z;

    double pre_loglikelihood_;
    bool start_training_;

    double *ptr_init_prob_t_;
    double *ptr_init_prob_e_;

};

#endif //HMM_EM_H
