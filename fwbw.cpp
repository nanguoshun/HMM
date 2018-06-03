//
// Created by  ngs on 01/06/2018.
//
#include "fwbw.h"

FB::FB() {

}
double FB::Forward(std::map<std::string, int> *ptr_x_corpus_map, std::vector<std::string> seq, std::pair<int, int> u_j,
                   HMMParameters parameters) {
    std::vector<double> pre_alpha;
    std::vector<double> current_alpha;
    //if alpha_u_1;
    if(u_j.second == 1){
        return (*parameters.ptr_t_)[0][u_j.first-1];
    }
    //calc the base value
    for (int v = 1; v <= parameters.num_of_state_ - 2; v++) {
        pre_alpha.push_back((*parameters.ptr_t_)[0][v-1]);
    }
    //iteration from position 2 to j;
    for (int t = 2; t <= u_j.second; t++) {
        //x_{j-1}
        int index = ptr_x_corpus_map->find(seq[t-2])->second;
        //calculation at the jth position
        if(t == u_j.second){
            double alpha_u_j = 0;
            for (int v = 1; v <= parameters.num_of_state_ - 2; v++) {
                double tranprob = (*parameters.ptr_t_)[v][u_j.first-1];
                double emprob = (*parameters.ptr_e_)[v-1][index];
                double alpha = pre_alpha[v-1];
                double alpha_v_u_j = tranprob * emprob * alpha;
                alpha_u_j += alpha_v_u_j;
            }
            return alpha_u_j;
        }else{
            //calculation at the intermediate position.
            for (int u = 1; u <= parameters.num_of_state_ - 2; u++) {
                double alpha_u_j = 0;
                for (int v = 1; v <= parameters.num_of_state_ - 2; v++) {
                    double tranprob = (*parameters.ptr_t_)[v][u-1];
                    double emprob = (*parameters.ptr_e_)[v-1][index];
                    double alpha = pre_alpha[v-1];
                    double alpha_v_u_j = tranprob * emprob * alpha;
                    alpha_u_j += alpha_v_u_j;
                }
                current_alpha.push_back(alpha_u_j);
            }
            pre_alpha.clear();
            pre_alpha = current_alpha;
            current_alpha.clear();
        }
    }
}

double FB::BackWard(std::map<std::string, int> *ptr_x_corpus_map, std::vector<std::string> seq, std::pair<int, int> u_j, HMMParameters parameters) {
    std::vector<double> next_beta;
    std::vector<double> currrent_beta;
    //calc the base value
    int size = seq.size();
    if(u_j.second == size){
        int index = ptr_x_corpus_map->find(seq[size-1])->second;
        double tranprob = (*parameters.ptr_t_)[u_j.first][parameters.num_of_state_-2];
        double emprob = (*parameters.ptr_e_)[u_j.first-1][index];
//        double value = (*parameters.ptr_t_)[u_j.first][parameters.num_of_state_-2] * (*parameters.ptr_e_)[u_j.first-1][index];
        double value = tranprob * emprob;
        return value;
    }
    int xn_index = ptr_x_corpus_map->find(seq[size-1])->second;
    for (int u = 1; u <= parameters.num_of_state_-2; ++u) {
        next_beta.push_back((*parameters.ptr_t_)[u][parameters.num_of_state_-2] * (*parameters.ptr_e_)[u-1][xn_index]);
    }
    //calc from the (size-1)th position to the jth position.
    //t indicates the position that starts from the 1, i.e. the sequence is 1, 2, 3.......
    for(int t = size-1; t>=u_j.second; t--){
        int index = ptr_x_corpus_map->find(seq[t-1])->second;
        if(t == u_j.second){
            double beta_u = 0;
            for(int v=1; v<=parameters.num_of_state_-2; v++){
                double tranprob = (*parameters.ptr_t_)[u_j.first][v-1];
                double emprob =  (*parameters.ptr_e_)[u_j.first-1][index];
                double beta =  next_beta[v-1];
                double beta_u_v = tranprob * emprob * beta;
//                double beta_u_v = (*parameters.ptr_t_)[u_j.first][v-1] * (*parameters.ptr_e_)[u_j.first-1][index] * next_beta[v-1];
                beta_u += beta_u_v;
            }
            return beta_u;
        }else{
            for(int u=1; u<=parameters.num_of_state_-2; u++){
                double beta_u = 0;
                for(int v=1; v<=parameters.num_of_state_-2; v++){
                    double tranprob = (*parameters.ptr_t_)[u][v-1];
                    double emprob = (*parameters.ptr_e_)[u-1][index];
                    double beta = next_beta[v-1];
//                    double beta_u_v = (*parameters.ptr_t_)[u][v-1] * (*parameters.ptr_e_)[u-1][index] * next_beta[v-1];
                    double beta_u_v = tranprob * emprob * beta;
                    beta_u += beta_u_v;
                }
                currrent_beta.push_back(beta_u);
            }
            next_beta.clear();
            next_beta = currrent_beta;
            currrent_beta.clear();
        }
    }
}

