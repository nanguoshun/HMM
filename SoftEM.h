//
// Created by  ngs on 01/06/2018.
//
#ifndef HMM_EEM_H
#define HMM_EEM_H

#include "common.h"
#include <vector>
#include "datasetmgr.h"
#include <cmath>
#include "fwbw.h"

class SoftEM{
public:
    SoftEM(const char* file_name, DatasetMgr *ptr_datamgr);
    ~SoftEM();
    void EStep();
    void MStep();
    void Normalize();
    void CalcUV(std::vector<std::string> seq, double Z_i);
    void CalcUO(std::vector<std::string> seq,double Z_i);
    double CalcU(std::vector<std::string> seq,int u, double Z_i);
    void Learning();
    bool IsIteration();
    double CalcPX(std::vector<std::string> seq);
    void GenerateSeqFromVector(std::vector<std::string> *ptr_vector, std::vector<std::vector<std::string>> *ptr_seq_vector);
    void RandomInitProb(double *ptr_prob_array, int array_size);
    void Init();
    void ResetCount();

private:
    DatasetMgr *ptr_datamgr_;
    std::set<std::string> *ptr_x_set_;
    std::vector<std::string> *ptr_train_x_vector_; //sequence is separated by a FLAG;
    std::vector<std::vector<std::string>> *ptr_training_seq_;

    FB *ptr_fwbw;
    size_t  number_of_x_;
    size_t  num_of_training_setence_;
    HMMParameters HMM_Parameters_;
    //std::vector<std::vector<double>> **pptr_alpha_;
    //std::vector<std::vector<double>> **pptr_beta_;
    std::map<std::string, int> *ptr_x_corpus_map_;
    std::vector<std::string> *ptr_x_corpus_;

    std::vector<double > *ptr_Z_;

    double pre_loglikelihood_;
    bool start_training_;

    double *ptr_init_prob_t_;
    double *ptr_init_prob_e_;
};

#endif //HMM_EEM_H
