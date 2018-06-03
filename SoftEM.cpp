//
// Created by  ngs on 01/06/2018.
//

#include "SoftEM.h"

SoftEM::SoftEM(const char *file_name, DatasetMgr *ptr_datamgr) {
    ptr_datamgr_ = ptr_datamgr;
    ptr_x_set_ = ptr_datamgr->GetTrainingXSet();
    number_of_x_ = ptr_datamgr->GetTrainingXSet()->size();
    ptr_train_x_vector_ = ptr_datamgr->GetTrainingXVector();
//    HMM_Parameters_.num_of_state_ = 4;//ptr_datamgr_->GetTagSet()->size() + 2;
 //   HMM_Parameters_.num_of_x_ = 3;//number_of_x_;
    HMM_Parameters_.num_of_state_ = ptr_datamgr_->GetTagSet()->size() + 2;
    HMM_Parameters_.num_of_x_ = number_of_x_;
    num_of_training_setence_ = ptr_datamgr->GetNumOfTrainingSeqs();
    ptr_training_seq_ = new std::vector<std::vector<std::string>>();
    GenerateSeqFromVector(ptr_train_x_vector_, ptr_training_seq_);
    //transition maxtrix
    int num_of_cloumn = HMM_Parameters_.num_of_state_ - 1;
    HMM_Parameters_.ptr_t_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_ - 1, std::vector<double>(HMM_Parameters_.num_of_state_ - 1, 1));
    //emission matrix
    HMM_Parameters_.ptr_e_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_ - 2, std::vector<double>(number_of_x_, 1));
    //transition maxtrix
    //HMM_Parameters_.ptr_t_next_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_ - 1, std::vector<double>(HMM_Parameters_.num_of_state_ - 1, 1));
    //emission matrix
    //HMM_Parameters_.ptr_e_next_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_ - 1, std::vector<double>(number_of_x_, 1));
    HMM_Parameters_.ptr_count_uv_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_ - 1, std::vector<double>(HMM_Parameters_.num_of_state_ - 1, 1));
    HMM_Parameters_.ptr_count_u_ = new std::vector<double>;
    HMM_Parameters_.ptr_count_uk_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_ - 2, std::vector<double>(number_of_x_, 1));
    ptr_fwbw = new FB();
    ptr_x_corpus_map_ = new std::map<std::string, int>;
    ptr_x_corpus_ = new std::vector<std::string>;
    ptr_Z_ = new std::vector<double>;
    ptr_init_prob_t_ = new double[HMM_Parameters_.num_of_state_ - 1];
    ptr_init_prob_e_ = new double[HMM_Parameters_.num_of_x_ - 2];
}

void SoftEM::RandomInitProb(double *ptr_prob_array, int array_size) {
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

void SoftEM::Init() {
    //double t[3][3] = {{0.3,0.7,0},{0.4,0.5,0.1},{0.6,0.2,0.2}};
    //double e[2][3] = {{0.3,0.4,0.3},{0.4,0.2,0.4}};
    for (int u = 0; u <= HMM_Parameters_.num_of_state_ - 2; u++) {
        //A, B, STOP
        //a_START, STOP = 0;
        if(u==0){
            RandomInitProb(ptr_init_prob_t_, HMM_Parameters_.num_of_state_ - 2);
        }else {
            RandomInitProb(ptr_init_prob_t_, HMM_Parameters_.num_of_state_ - 1);
        }
        for (int v = 1; v <= HMM_Parameters_.num_of_state_ - 1; v++) {
            //double prob = (double) 1 / (double) (HMM_Parameters_.num_of_state_ - 1);
            if(v == HMM_Parameters_.num_of_state_-1 && u==0){
                (*HMM_Parameters_.ptr_t_)[u][v - 1] = 0;
            }else{
                (*HMM_Parameters_.ptr_t_)[u][v - 1] = ptr_init_prob_t_[v-1];
//                (*HMM_Parameters_.ptr_t_)[u][v - 1] = t[u][v-1];
            }
            //(*HMM_Parameters_.ptr_t_next_)[u][v-1] = 0;
            //std::cout << "ptr_a_" <<u<<","<<v-1<<"="<<(*HMM_Parameters_.ptr_t_)[u][v - 1]<<std::endl;
        }
    }
    for (int v = 1; v <= HMM_Parameters_.num_of_state_ - 2; v++) {
        //init emission prob
        RandomInitProb(ptr_init_prob_e_, HMM_Parameters_.num_of_x_);
        for (int k = 0; k < number_of_x_; k++) {
            double prob = (double) 1 / (double) number_of_x_;
            (*HMM_Parameters_.ptr_e_)[v-1][k] = ptr_init_prob_e_[k];
//            (*HMM_Parameters_.ptr_e_)[v-1][k] = e[v-1][k];
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

SoftEM::~SoftEM() {
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
}

double SoftEM::CalcPX(std::vector<std::string> seq) {
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

double SoftEM::CalcU(std::vector<std::string> seq, int u, double Z_i) {
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
void SoftEM::CalcUV(std::vector<std::string> seq, double Z_i) {
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
void SoftEM::CalcUO(std::vector<std::string> seq, double Z_i) {
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

bool SoftEM::IsIteration() {
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

void SoftEM::EStep() {
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

void SoftEM::MStep() {
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

void SoftEM::Normalize() {
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
    for (int u = 0; u <= HMM_Parameters_.num_of_state_ - 2; u++) {
        double uv = 0;
        for (int v = 1; v <= HMM_Parameters_.num_of_state_ - 1; v++) {
            uv += (*HMM_Parameters_.ptr_t_)[u][v - 1];
        }
        //std::cout << "UV: the sum of the " <<u<< "th row is: "<<uv<<std::endl;
    }
}
void SoftEM::Learning() {
    Init();
    while (IsIteration()) {
        EStep();
        MStep();
    }
}

void SoftEM::ResetCount() {
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

void SoftEM::GenerateSeqFromVector(std::vector<std::string> *ptr_vector,
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