//
// Created by  ngs on 01/06/2018.
//

#include "em.h"
EM::EM(DatasetMgr *ptr_datamgr) {
    ptr_datamgr_ = ptr_datamgr;
    ptr_x_set_ = ptr_datamgr->GetTrainingXSet();
    number_of_x_ = ptr_datamgr->GetTrainingXSet()->size();
    ptr_train_x_vector_ = ptr_datamgr->GetTrainingXVector();
    HMM_Parameters_.num_of_state_ = ptr_datamgr_->GetTagSet()->size() + 2;
    HMM_Parameters_.num_of_x_ = number_of_x_;
    
#ifdef TEST_MODE
    HMM_Parameters_.num_of_state_ = 4;//ptr_datamgr_->GetTagSet()->size() + 2;
    HMM_Parameters_.num_of_x_ = 3;//number_of_x_;
#endif
    num_of_training_setence_ = ptr_datamgr->GetNumOfTrainingSeqs();
    ptr_training_seq_ = new std::vector<std::vector<std::string>>();
    GenerateSeqFromVector(ptr_train_x_vector_, ptr_training_seq_);
    //transition maxtrix
    HMM_Parameters_.ptr_t_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_ - 1, std::vector<double>(HMM_Parameters_.num_of_state_ - 1, 1));
    //emission matrix
    HMM_Parameters_.ptr_e_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_ - 2, std::vector<double>(number_of_x_, 1));
    HMM_Parameters_.ptr_count_uv_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_ - 1, std::vector<double>(HMM_Parameters_.num_of_state_ - 1, 1));
    HMM_Parameters_.ptr_count_u_ = new std::vector<double>;
    HMM_Parameters_.ptr_count_uk_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_ - 2, std::vector<double>(number_of_x_, 1));
    //to record the previous node with maximized prob.
    ptr_path_matrix_ = new std::vector<std::vector<int>>(HMM_Parameters_.num_of_state_ - 2, std::vector<int>(number_of_x_, 1));
    //the path
    ptr_path_ = new std::vector<int>(number_of_x_);
    ptr_fwbw = new FB();
    ptr_x_corpus_map_ = new std::map<std::string, int>;
    ptr_x_corpus_ = new std::vector<std::string>;
    ptr_Z_ = new std::vector<double>;
    ptr_init_prob_t_ = new double[HMM_Parameters_.num_of_state_ - 1];
    ptr_init_prob_e_ = new double[HMM_Parameters_.num_of_x_ - 2];
}

void EM::RandomInitProb(double *ptr_prob_array, int array_size) {
    double sum = 0;
    for (int i = 0; i < array_size; i++) {
        ptr_prob_array[i] = rand() % RAND_MAX_NUM;
    }
    for (int i = 0; i < array_size; i++) {
        sum += ptr_prob_array[i];
    }
    for (int i = 0; i < array_size; i++) {
        ptr_prob_array[i] = ptr_prob_array[i] / sum;
        // std::cout<<ptr_prob_array[i]<<std::endl;
    }
}

void EM::Init() {
#ifdef TEST_MODE
    double t[3][3] = {{0.3, 0.7, 0}, {0.4, 0.5, 0.1}, {0.6, 0.2, 0.2}};
    double e[2][3] = {{0.3, 0.4, 0.3}, {0.4, 0.2, 0.4}};
#endif
    for (int u = 0; u <= HMM_Parameters_.num_of_state_ - 2; u++) {
        //A, B, STOP
        //a_START, STOP = 0;
        if (u == 0) {
            RandomInitProb(ptr_init_prob_t_, HMM_Parameters_.num_of_state_ - 2);
        } else {
            RandomInitProb(ptr_init_prob_t_, HMM_Parameters_.num_of_state_ - 1);
        }
        for (int v = 1; v <= HMM_Parameters_.num_of_state_ - 1; v++) {
            //double prob = (double) 1 / (double) (HMM_Parameters_.num_of_state_ - 1);
            if (v == HMM_Parameters_.num_of_state_ - 1 && u == 0) {
                (*HMM_Parameters_.ptr_t_)[u][v - 1] = 0;
            } else {
#ifndef TEST_MODE
                (*HMM_Parameters_.ptr_t_)[u][v - 1] = ptr_init_prob_t_[v-1];
#endif
#ifdef TEST_MODE
                (*HMM_Parameters_.ptr_t_)[u][v - 1] = t[u][v - 1];
#endif
            }
            //(*HMM_Parameters_.ptr_t_next_)[u][v-1] = 0;
            //std::cout << "ptr_a_" <<u<<","<<v-1<<"="<<(*HMM_Parameters_.ptr_t_)[u][v - 1]<<std::endl;
        }
    }
    for (int v = 1; v <= HMM_Parameters_.num_of_state_ - 2; v++) {
        //init emission prob
        RandomInitProb(ptr_init_prob_e_, HMM_Parameters_.num_of_x_);
        for (int k = 0; k < number_of_x_; k++) {
#ifndef TEST_MODE
            (*HMM_Parameters_.ptr_e_)[v-1][k] = ptr_init_prob_e_[k];
#endif
#ifdef TEST_MODE
            (*HMM_Parameters_.ptr_e_)[v - 1][k] = e[v - 1][k];
#endif
            //(*HMM_Parameters_.ptr_e_next_)[u][k] = 0;
            // std::cout << "ptr_b_" <<v<<","<<k<<"="<<(*HMM_Parameters_.ptr_e_)[v-1][k]<<std::endl;
        }
    }
    //to simplify the learning, we use the training x set as corpus.
    int index = 0;
    for (std::set<std::string>::iterator it = ptr_x_set_->begin(); it != ptr_x_set_->end(); it++) {
        std::cout << (*it) << std::endl;
        ptr_x_corpus_map_->insert(std::make_pair((*it), index));
        ptr_x_corpus_->push_back((*it));
        index++;
    }
    pre_loglikelihood_ = 0;
    start_training_ = false;
}

EM::~EM() {
    delete ptr_training_seq_;
    delete HMM_Parameters_.ptr_t_;
    delete HMM_Parameters_.ptr_e_;
    //delete HMM_Parameters_.ptr_t_next_;
    //delete HMM_Parameters_.ptr_e_next_;
    delete HMM_Parameters_.ptr_count_uv_;
    delete HMM_Parameters_.ptr_count_u_;
    delete HMM_Parameters_.ptr_count_uk_;
    delete ptr_fwbw;
    delete ptr_x_corpus_;
    delete ptr_x_corpus_map_;
    delete ptr_Z_;
    delete ptr_init_prob_t_;
    delete ptr_init_prob_e_;
    delete ptr_path_matrix_;
    delete ptr_path_;
}

double EM::CalcPX(std::vector<std::string> seq) {
    double px = 0;
    int size = seq.size();
    //alpha_u_1, here 1 indicates the start position.
    for(int u=1; u<=HMM_Parameters_.num_of_state_-2; u++){
        double alpha_u_j = ptr_fwbw->Forward(ptr_x_corpus_map_, seq, std::make_pair(u,size), HMM_Parameters_);
        double beta_u_j  = ptr_fwbw->BackWard(ptr_x_corpus_map_, seq, std::make_pair(u,size), HMM_Parameters_);
        double value = alpha_u_j * beta_u_j;
        px += value;
    }
    std::cout << "pix "<<px << std::endl;
    return px;
}

double EM::CalcU(std::vector<std::string> seq, int u, double Z_i) {
    //if it is y_0, then the P(y_0 = START) = 1, hence COUNT(START) = 1;
    if (u == 0) {
        return 1;
    } else {
        //return COUNT(U);
        int size = seq.size();
        double numerator = 0;
        double count_u_j = 0;
        for (int j = 1; j <= size; j++) {
            double alpha_u_j = ptr_fwbw->Forward(ptr_x_corpus_map_, seq, std::make_pair(u, j), HMM_Parameters_);
            double beta_u_j = ptr_fwbw->BackWard(ptr_x_corpus_map_, seq, std::make_pair(u, j), HMM_Parameters_);
            count_u_j = alpha_u_j * beta_u_j;
            numerator += count_u_j;
        }
        //std::cout << "count_u of "<<u<<" th state is "<<numerator/Z_i<<std::endl;
        return numerator / Z_i;
    }
}
/**
 * calc the count(U,V) for a sequence.
 * @param seq : targeting sequence
 * @param Z_i : denominator
 */
void EM::CalcUV(std::vector<std::string> seq, double Z_i) {
    for (int u = 0; u <= HMM_Parameters_.num_of_state_ - 2; u++) {
        for (int v = 1; v <= HMM_Parameters_.num_of_state_ - 1; v++) {
            double numerator = 0;
            //calc COUNT(START, V), actually,
            if (u == 0) {
                //if it is COUNT(START, STOP), then the count should be 0;
                if (v == HMM_Parameters_.num_of_state_ - 1) {
                    numerator = 0;
                } else {
                    //calc COUNT(STAR, V)
                    double beta_v_1 = ptr_fwbw->BackWard(ptr_x_corpus_map_, seq, std::make_pair(v, 1),HMM_Parameters_);
                    numerator = (*HMM_Parameters_.ptr_t_)[u][v - 1] * beta_v_1;
                }
            } else {
                int size = seq.size();
                //COUNT(u, STOP)
                if (v == HMM_Parameters_.num_of_state_ - 1) {
                    int index = ptr_x_corpus_map_->find(seq[size - 1])->second;
                    double alpha_u_n = ptr_fwbw->Forward(ptr_x_corpus_map_, seq, std::make_pair(u, size), HMM_Parameters_);
                    numerator = alpha_u_n * (*HMM_Parameters_.ptr_t_)[u][v - 1] * (*HMM_Parameters_.ptr_e_)[u-1][index];
                } else {
                    //COUNT(U,V)
                    double count_u_v_j = 0;
                    for (int j = 1; j <= size; j++) {
                        int index = ptr_x_corpus_map_->find(seq[j-1])->second;
                        double alpha_u_j = ptr_fwbw->Forward(ptr_x_corpus_map_, seq, std::make_pair(u, j), HMM_Parameters_);
                        double beta_u_j_plus_1 = ptr_fwbw->BackWard(ptr_x_corpus_map_, seq, std::make_pair(u, j + 1), HMM_Parameters_);
                        count_u_v_j = alpha_u_j * (*HMM_Parameters_.ptr_t_)[u][v - 1] * (*HMM_Parameters_.ptr_e_)[u-1][index] * beta_u_j_plus_1;
                        numerator += count_u_v_j;
                    }
                }
            }
            (*HMM_Parameters_.ptr_count_uv_)[u][v - 1] += numerator/Z_i;
        }
    }
}
/**
 * calc the COUNT(U->o) for a sequence.
 * @param seq
 * @param Z_i
 */
void EM::CalcUO(std::vector<std::string> seq, double Z_i) {
    int size = seq.size();
    for(int u = 1; u<= HMM_Parameters_.num_of_state_-2; u++){
        double count_u_k = 0;
        for(int k = 0; k<=HMM_Parameters_.num_of_x_-1; k++){
            double numerator = 0;
            for(int j=1; j<=size; j++){
                //std::cout << (*ptr_x_corpus_)[k] << std::endl;
                if(seq[j-1] == (*ptr_x_corpus_)[k]){
                    double alpha_u_j = ptr_fwbw->Forward(ptr_x_corpus_map_, seq, std::make_pair(u, j), HMM_Parameters_);
                    double beta_u_j  = ptr_fwbw->BackWard(ptr_x_corpus_map_, seq, std::make_pair(u, j), HMM_Parameters_);
                    count_u_k = alpha_u_j * beta_u_j;
                    numerator +=count_u_k;
                }
            }
            (*HMM_Parameters_.ptr_count_uk_)[u-1][k] += numerator/Z_i;
        }
    }
}

bool EM::IsIteration() {
    double loglikelihood = 0;
    ResetCount();
    ptr_Z_->clear();
    for(std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin(); it != ptr_training_seq_->end(); it++) {
        double z = CalcPX((*it));
        ptr_Z_->push_back(z);
        loglikelihood += log(z);
    }
    std::cout << "loglikelihood is" <<loglikelihood<<std::endl;
    return true;
}

std::pair<double,int> EM::Viterbi(std::vector<std::string> seq) {
    std::vector<double> pi_pre_vk;
    std::vector<double> pi_vk;
    int seq_size = seq.size();
    int finalnode = 0;
    //calc pi_vk
    for(int k=0; k<=seq_size; k++){
        //base case
        if(k==0){
            int index = ptr_x_corpus_map_->find(seq[k])->second;
            for(int v=1; v<=HMM_Parameters_.num_of_state_-2; v++){
                //pi(0, v), if v = start, then the vale is 1, otherwise 0;
                double tranprob = (*HMM_Parameters_.ptr_t_)[0][v-1];
                double emprob = (*HMM_Parameters_.ptr_e_)[v-1][index];
                double pi_v = tranprob * emprob;
                pi_pre_vk.push_back(pi_v);
                (*ptr_path_matrix_)[v-1][0] = 0;
            }
            continue;
        }
        if(k == seq_size){
            double max_pi_v = 0;
            for(int u=1; u<= HMM_Parameters_.num_of_state_-2; u++){
                double tranprob = (*HMM_Parameters_.ptr_t_)[u][HMM_Parameters_.num_of_state_-2];
                double vk = pi_pre_vk[u-1];
                double pi_v = vk * tranprob;
                if(pi_v >= max_pi_v){
                    max_pi_v = pi_v;
                    finalnode = u;
                }
            }
            return std::make_pair(max_pi_v, finalnode);
        }else{
            int index = ptr_x_corpus_map_->find(seq[k])->second;
            for(int v=1;v <= HMM_Parameters_.num_of_state_-2; v++){
                double max_pi_u = 0;
                for(int u=1; u<= HMM_Parameters_.num_of_state_-2; u++){
                    double tranprob = (*HMM_Parameters_.ptr_t_)[u][v-1];
                    double emprob = (*HMM_Parameters_.ptr_e_)[v-1][index];
                    double vk = pi_pre_vk[u-1];
                    double pi_u_v = tranprob * emprob * vk;
                    if(pi_u_v >= max_pi_u){
                        max_pi_u = pi_u_v;
                        (*ptr_path_matrix_)[v-1][k] = u;
                    }
                }
                pi_vk.push_back(max_pi_u);
            }
            pi_pre_vk.clear();
            pi_pre_vk = pi_vk;
            pi_vk.clear();
        }
    }
}

void EM::BackTracking(std::pair<double,int> viterbi_result) {
    double final_node = viterbi_result.second;
    double pre_node = final_node;
    //the final node is known
    (*ptr_path_)[HMM_Parameters_.num_of_x_-1] = final_node;
    //backtracking the rest of the node.
    for(int k = HMM_Parameters_.num_of_x_-2; k>=0; k--){
        (*ptr_path_)[k] = (*ptr_path_matrix_)[pre_node-1][k+1];
        pre_node = (*ptr_path_)[k];
    }
    for(int k=0; k<HMM_Parameters_.num_of_x_; k++){
        std::cout << (*ptr_path_)[k] <<std::endl;
    }
}

void EM::HardEStep() {
    for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin();
         it != ptr_training_seq_->end(); it++) {
        std::pair<double,int> viterbi_result = Viterbi((*it));
        BackTracking(viterbi_result);
    }
}

void EM::SoftEStep() {
    HMM_Parameters_.ptr_count_u_->clear();
    int z_index = 0;
    double count[HMM_Parameters_.num_of_state_-1];
    memset(count, 0, sizeof(double)*(HMM_Parameters_.num_of_state_-1));
    for(std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin(); it != ptr_training_seq_->end(); it++) {
        for(int u=0; u<=HMM_Parameters_.num_of_state_-2; u++){
            count[u] += CalcU((*it),u,(*ptr_Z_)[z_index]);
        }
        CalcUV((*it),(*ptr_Z_)[z_index]);
        CalcUO((*it),(*ptr_Z_)[z_index]);
        z_index ++;
    }
    for(int u=0; u<=HMM_Parameters_.num_of_state_-2; u++){
        HMM_Parameters_.ptr_count_u_->push_back(count[u]);
    }
}

void EM::MStep() {
    for (int u = 0; u <= HMM_Parameters_.num_of_state_ - 2; u++) {
        double count_u = (*HMM_Parameters_.ptr_count_u_)[u];
        for (int v = 1; v <= HMM_Parameters_.num_of_state_ - 1; v++) {
            double count_uv = (*HMM_Parameters_.ptr_count_uv_)[u][v - 1];
            (*HMM_Parameters_.ptr_t_)[u][v - 1] = count_uv / count_u;
//            std::cout << "<u, v>: <" << u << "," << v - 1 << ">: " << (*HMM_Parameters_.ptr_t_)[u][v - 1] << std::endl;

        }
    }
    for (int u = 1; u <= HMM_Parameters_.num_of_state_ - 2; u++) {
        double count_u = (*HMM_Parameters_.ptr_count_u_)[u];
        for (int k = 0; k < HMM_Parameters_.num_of_x_; k++) {
            double count_uk = (*HMM_Parameters_.ptr_count_uk_)[u - 1][k];
            (*HMM_Parameters_.ptr_e_)[u - 1][k] = count_uk / count_u;
//            std::cout << "<u, k>: <" << u << "," << k << ">: " << (*HMM_Parameters_.ptr_e_)[u - 1][k] << std::endl;
        }
    }
    Normalize();
}

void EM::Normalize() {
    std::vector<double> u_vector;
    for (int u = 0; u <= HMM_Parameters_.num_of_state_ - 2; u++) {
        double uv = 0;
        for (int v = 1; v <= HMM_Parameters_.num_of_state_ - 1; v++){
            uv +=(*HMM_Parameters_.ptr_t_)[u][v - 1];
        }
        u_vector.push_back(uv);
        //std::cout << "UV: the sum of the " <<u<< "th row is: "<<uv<<std::endl;
    }
    for (int u = 1; u <= HMM_Parameters_.num_of_state_ - 2; u++) {
        double uk = 0;
        for (int k = 0; k < HMM_Parameters_.num_of_x_; k++) {
            uk += (*HMM_Parameters_.ptr_e_)[u - 1][k];
        }
        //std::cout << "UK: the sum of the " <<u-1<< "th row is: "<<uk<<std::endl;
    }
    for (int u = 0; u <= HMM_Parameters_.num_of_state_ - 2; u++) {
        for (int v = 1; v <= HMM_Parameters_.num_of_state_ - 1; v++){
            (*HMM_Parameters_.ptr_t_)[u][v - 1] = (*HMM_Parameters_.ptr_t_)[u][v - 1]/u_vector[u];
        }
    }
    /*
    for (int u = 0; u <= HMM_Parameters_.num_of_state_ - 2; u++) {
        double uv = 0;
        for (int v = 1; v <= HMM_Parameters_.num_of_state_ - 1; v++) {
            uv += (*HMM_Parameters_.ptr_t_)[u][v - 1];
        }
        //std::cout << "UV: the sum of the " <<u<< "th row is: "<<uv<<std::endl;
    }
     */
}
void EM::Learning(bool is_soft_em) {
    Init();
    while (IsIteration()) {
        if(is_soft_em){
            SoftEStep();
        } else{
            HardEStep();
        }
        MStep();
    }
}

void EM::ResetCount() {
    for (int u = 0; u <= HMM_Parameters_.num_of_state_ - 2; u++) {
        for (int v = 1; v <= HMM_Parameters_.num_of_state_ - 1; v++) {
            (*HMM_Parameters_.ptr_count_uv_)[u][v - 1] = 0;
        }
    }
    for (int u = 1; u <= HMM_Parameters_.num_of_state_ - 2; u++){
        for (int k = 0; k < HMM_Parameters_.num_of_x_; k++) {
            (*HMM_Parameters_.ptr_count_uk_)[u - 1][k] = 0;
        }
    }
}

void EM::GenerateSeqFromVector(std::vector<std::string> *ptr_vector,
                                   std::vector<std::vector<std::string>> *ptr_seq_vector) {
    std::vector<std::string> seq;
    for (std::vector<std::string>::iterator it = ptr_vector->begin(); it != ptr_vector->end(); it++) {
        //std::cout<<*it<<std::endl;
        if (*it == SPERATOR_FLAG) {
            ptr_seq_vector->push_back(seq);
            seq.clear();
            continue;
        } else {
            seq.push_back(*it);
            //do not forget the last seq which doesn't contain a SPEARATOR_FLAG at the end.
            if (it == (ptr_vector->end() - 1)) {
                ptr_seq_vector->push_back(seq);
            }
        }
    }
}