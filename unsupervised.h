//
// Created by  ngs on 05/05/2018.
//

#ifndef HMM_UNSUPERVISED_H
#define HMM_UNSUPERVISED_H

#include <tuple>
#include <string>
#include <map>
#include <math.h>
#include "datasetmgr.h"

class Learning{

public:
    explicit Learning(const char* file_name, DatasetMgr *ptr_datamgr);
    ~Learning();
    void Init();
    std::pair<double, double> Forward(const std::vector<std::string> sequence, size_t seq_no);
    void Backward(const std::vector<std::string> sequence, size_t seq_no);
    void BaumWelch();
    double CalcGamma(size_t seq_no, size_t t, size_t i);
    double CalcXi(size_t seq_no, size_t t, size_t i, size_t j);
    bool IsIteration(size_t iteration_no);
    void CalcPi();
    void CalcA();
    void CalcB();
    void UpdateParameters();
    void StartTraining();
    void GenerateSeqFromVector(std::vector<std::string> *ptr_vector, std::vector<std::vector<std::string>> *ptr_seq_vector);

private:
    size_t  number_of_state_;
    size_t  number_of_x_;
    size_t  num_of_training_setence_;
    double  current_loglikelihood_;
    double  pre_loglikelihood_;
    std::map<std::string, double > *ptr_state_to_state_prob_map_;
    std::vector<std::string> *ptr_train_x_vector_; //sequence is separated by a FLAG;
    std::set<std::string> *ptr_state_set_;
    std::vector<std::vector<std::string>> *ptr_training_seq_;
    std::set<std::string> *ptr_x_set_;
    std::vector<std::string> *ptr_x_corpus_;


    //parameters of an HMM model.
    std::vector<double> *ptr_pi_;
    std::vector<std::vector<double >> *ptr_b_;
    std::vector<std::vector<double >> *ptr_a_;

    //updated value for the parameters of an HMM model during EM-E step.
    std::vector<double> *ptr_next_pi_;
    std::vector<std::vector<double >> *ptr_next_b_;
    std::vector<std::vector<double >> *ptr_next_a_;

    //we have multiple observation sequences. pptr_alpha_[a][t,i,p]: the element a,t,i,p indicates the sequence no, position, state, value.
    std::vector<std::vector<double>> **pptr_alpha_;
    std::vector<std::vector<double>> **pptr_beta_;
    std::vector<double > *ptr_PO_; //P(O|lamda}
    //std::vector<std::vector<double>> *ptr_alpha_;
    //std::vector<std::vector<double >> *ptr_beta_;

    DatasetMgr *ptr_datamgr_;

};

#endif //HMM_UNSUPERVISED_H
