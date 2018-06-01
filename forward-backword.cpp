//
// Created by  ngs on 12/05/2018.
//

#include "forward-backword.h"

ForwardBackward::ForwardBackward() {

}

double ForwardBackward::ForwardResult(std::map<std::string, int> *ptr_x_corpus_map, std::vector<std::string> seq, std::pair<int, int> u_j, HMMParameters parameters) {
    int u = u_j.first;
    int j = u_j.second;
    int num_of_state = parameters.num_of_state_;

    std::vector<std::vector<double>> *ptr_t = parameters.ptr_t_;
    std::vector<std::vector<double>> *ptr_e = parameters.ptr_e_;
    /**
     * below code is for observation only
     */
    int num_of_x = parameters.num_of_x_;
    std::vector<std::vector<double>> tr;
    std::vector<double> trtr;
    std::vector<std::vector<double>> e;
    std::vector<double> ee;
    for (int uuu = 0; uuu <= num_of_state-2; uuu++) {
        //A, B, STOP
        trtr.clear();
        for (int v = 1; v <= num_of_state-1; v++) {
            trtr.push_back((*ptr_t)[uuu][v-1]);
            //std::cout << "ptr_t_" <<uuu<<","<<v<<"="<<(*ptr_t)[uuu][v]<<std::endl;
        }
        tr.push_back(trtr);
        if(uuu>=1){
            ee.clear();
            for (int k = 0; k < num_of_x; k++) {
                ee.push_back((*ptr_e)[uuu][k]);
                //std::cout << "ptr_e_" <<uuu<<","<<k<<"="<<(*ptr_e)[uuu][k]<<std::endl;
            }
        e.push_back(ee);
        }
    }
    //above code is for observation only
    std::vector<double> alpha_u_j;
    std::vector<double> alpha_u_j_next;
    if (j == 1) {
        // for alpha_u(1), return a_START_u, in the transition matrix, the coordinator is (0, u-1);
        return (*ptr_t)[0][u-1];
    }
    for (int v = 1; v <= num_of_state - 2; v++) { // init a_v(1) with equal prob, i.e, prob= 1/num_of_state
        alpha_u_j.push_back((*ptr_t)[0][v-1]);  // from P(Y_1 = u|Y_START), u \in {A, B, C}, P(Y_STOP|Y_START) is meaningless
    }
    for (int t = 2; t <= j; t++) {
        std::string str = seq[t-2];
        int index = ptr_x_corpus_map->find(str)->second;
        if (t == j) {
            double a = 0;
            for (int v = 1; v <= num_of_state-2; v++) {
                double alpha= alpha_u_j[v-1];
                double tuv= (*ptr_t)[v][u-1];
                double ev = (*ptr_e)[v][index];
                a += alpha * tuv * ev;
//              a += alpha_u_j[v-1] * (*ptr_t)[v][u] * (*ptr_e)[v][index];
            }
            return a;
        } else {  // calc the a_u(j) iteratively
            for (int uu = 1; uu <= num_of_state-2; uu++) {
                double a = 0;
                for (int v = 1; v <= num_of_state-2; v++) {
                    double tran_prob = (*ptr_t)[v][uu-1];
                    double em_prob = (*ptr_e)[v][index];
                    a += alpha_u_j[v-1] * tran_prob * em_prob;
                    if(em_prob == 0){
                        std::cout << "em_prob equals 0"<<std::endl;
                    }
                }
                alpha_u_j_next.push_back(a);
            }
            alpha_u_j.clear();
            alpha_u_j = alpha_u_j_next;
            alpha_u_j_next.clear();
        }
    }
}

double ForwardBackward::BackwardResult(std::map<std::string, int> *ptr_x_corpus_map, std::vector<std::string> seq, std::pair<int, int> u_j, HMMParameters parameters) {
    int u = u_j.first;
    int j = u_j.second;
    int num_of_state = parameters.num_of_state_;
    std::vector<std::vector<double>> *ptr_t = parameters.ptr_t_;
    std::vector<std::vector<double>> *ptr_e = parameters.ptr_e_;
    std::vector<double> beta_u_j_next;
    std::vector<double> beta_u_j;
    /**
      * below code is for observation only
      */
    int num_of_x = parameters.num_of_x_;
    std::vector<std::vector<double>> tr;
    std::vector<double> trtr;
    std::vector<std::vector<double>> e;
    std::vector<double> ee;
    for (int uuu = 0; uuu <= num_of_state-2; uuu++) {
        //A, B, STOP
        trtr.clear();
        for (int v = 1; v <= num_of_state-1; v++) {
            trtr.push_back((*ptr_t)[uuu][v-1]);
            //std::cout << "ptr_a_" <<i<<","<<j<<"="<<(*ptr_a_)[i][j]<<std::endl;
        }
        tr.push_back(trtr);
        //init emission prob
        if(uuu>=1){
            ee.clear();
            for (int k = 0; k < num_of_x; k++) {
                ee.push_back((*ptr_e)[uuu][k]);
                // std::cout << "ptr_b_" <<i<<","<<k<<"="<<(*ptr_b_)[i][k]<<std::endl;
            }
            e.push_back(ee);
        }
    }
    //above code is for observation only

    if(1 == seq.size()){
        //\beta_v(1) = P(x_1 | Y_1 =v) = b_v(x_1)
        int index = ptr_x_corpus_map->find(seq[0])->second;
        return (*ptr_e)[u][index];
    }
    int last_word = ptr_x_corpus_map->find(seq[seq.size()-1])->second;
    if(j == seq.size()){
        double result = (*ptr_t)[u][num_of_state-2] * (*ptr_e)[u][last_word];
        return result;
    }
    //init \beta_u(x_n)
    for(int v=1; v<=num_of_state-2; v++){
        //\beta_u_(x_n) = a_{u,STOP} * b_u(x_n)
        double beta_u_xn = (*ptr_t)[u][num_of_state-2] * (*ptr_e)[u][last_word];
        beta_u_j_next.push_back(beta_u_xn);
    }
    //from the end to the beginning.
    for(int t = seq.size()-1; t>=j; t--){
        int index = ptr_x_corpus_map->find(seq[t])->second;
        if(t==j){
            double b = 0;
            for(int v=1; v<= num_of_state-2; v++){
                b += (*ptr_t)[u][v-1] * (*ptr_e)[u][index] * beta_u_j_next[v-1];
            }
            return b;
        } else{
            for(int uu=1; uu<= num_of_state-2; uu++){
                double b = 0;
                for(int v=1; v<= num_of_state-2; v++){
                    b += (*ptr_t)[uu][v-1] * (*ptr_e)[uu][index] * beta_u_j_next[v-1];
                }
                beta_u_j.push_back(b);
            }
            beta_u_j_next.clear();
            beta_u_j_next = beta_u_j;
            beta_u_j.clear();
        }
    }
}