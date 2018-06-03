//
// Created by  ngs on 12/05/2018.
//

#include "em.h"
#include <time.h>

EM::EM(const char* file_name, DatasetMgr *ptr_datamgr) {
    ptr_datamgr_ = ptr_datamgr;
    ptr_x_set_ = ptr_datamgr->GetTrainingXSet();
    number_of_x_ = ptr_datamgr->GetTrainingXSet()->size();
    ptr_train_x_vector_ = ptr_datamgr->GetTrainingXVector();
    HMM_Parameters_.num_of_state_ = ptr_datamgr_->GetTagSet()->size() + 2;
    HMM_Parameters_.num_of_x_ = number_of_x_;
    num_of_training_setence_ = ptr_datamgr->GetNumOfTrainingSeqs();
    ptr_training_seq_ = new std::vector<std::vector<std::string>>();
    GenerateSeqFromVector(ptr_train_x_vector_, ptr_training_seq_);
    //transition maxtrix
    int num_of_cloumn = HMM_Parameters_.num_of_state_-1;
    HMM_Parameters_.ptr_t_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_-1, std::vector<double>(HMM_Parameters_.num_of_state_-1, 1));
    //emission matrix
    HMM_Parameters_.ptr_e_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_-1, std::vector<double>(number_of_x_, 1));
    //transition maxtrix
    HMM_Parameters_.ptr_t_next_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_-1, std::vector<double>(HMM_Parameters_.num_of_state_-1, 1));
    //emission matrix
    HMM_Parameters_.ptr_e_next_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_-1, std::vector<double>(number_of_x_, 1));

    HMM_Parameters_.ptr_count_uv_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_-1, std::vector<double>(HMM_Parameters_.num_of_state_-1, 1));
    HMM_Parameters_.ptr_count_u_ =  new std::vector<double>;
    HMM_Parameters_.ptr_count_uk_ = new std::vector<std::vector<double>>(HMM_Parameters_.num_of_state_-1, std::vector<double>(number_of_x_, 1));
    ptr_fwbw = new ForwardBackward();
    ptr_x_corpus_map_ = new std::map<std::string, int>;
    ptr_x_corpus_ = new std::vector<std::string>;
    ptr_Z = new std::vector<double>;
    ptr_init_prob_t_ = new double[HMM_Parameters_.num_of_state_-1];
    ptr_init_prob_e_ = new double[HMM_Parameters_.num_of_x_-1];
}

EM::~EM() {
    delete ptr_training_seq_;
    delete HMM_Parameters_.ptr_t_;
    delete HMM_Parameters_.ptr_e_;
    delete HMM_Parameters_.ptr_t_next_;
    delete HMM_Parameters_.ptr_e_next_;
    delete HMM_Parameters_.ptr_count_uv_;
    delete HMM_Parameters_.ptr_count_u_;
    delete HMM_Parameters_.ptr_count_uk_;
    delete ptr_fwbw;
    delete ptr_x_corpus_;
    delete ptr_x_corpus_map_;
    delete ptr_Z;
    delete ptr_init_prob_t_;
    delete ptr_init_prob_e_;
}

void EM::RandomInitProb(double *ptr_prob_array, int array_size) {
    double sum = 0;
    for(int i =0; i< array_size; i++){
        ptr_prob_array[i] = rand() % RAND_MAX_NUM;
    }
    for(int i=0; i<array_size;i++){
        sum+=ptr_prob_array[i];
    }
    for(int i=0; i<array_size;i++){
        ptr_prob_array[i] = ptr_prob_array[i] / sum;
       // std::cout<<ptr_prob_array[i]<<std::endl;
    }
}

void EM::Init() {
    //START, A, B, C, STOP
    // we simply set state number as the one in tagged sequences. A more practical solution is expected to be developed.
    //init transition prob
    //start == 0
    //a_0_1, a_y_n_stop,totally (num_of_state - 1);  START, A, B
    RandomInitProb(ptr_init_prob_t_, HMM_Parameters_.num_of_state_-1);
    RandomInitProb(ptr_init_prob_e_,HMM_Parameters_.num_of_x_);
    for (int u = 0; u <= HMM_Parameters_.num_of_state_-2; u++) {
        //A, B, STOP
        for (int v = 1; v <= HMM_Parameters_.num_of_state_-1; v++) {
            double prob = (double) 1 / (double) (HMM_Parameters_.num_of_state_ - 1);
            (*HMM_Parameters_.ptr_t_)[u][v-1] = ptr_init_prob_t_[u];
            //(*HMM_Parameters_.ptr_t_next_)[u][v-1] = 0;
            //std::cout << "ptr_a_" <<i<<","<<j<<"="<<(*ptr_a_)[i][j]<<std::endl;
        }
        //init emission prob
        if(u>=1){
            for (int k = 0; k < number_of_x_; k++) {
                double prob = (double) 1 / (double) number_of_x_;
                (*HMM_Parameters_.ptr_e_)[u][k] = ptr_init_prob_e_[k];
                //(*HMM_Parameters_.ptr_e_next_)[u][k] = 0;
                // std::cout << "ptr_b_" <<i<<","<<k<<"="<<(*ptr_b_)[i][k]<<std::endl;
            }
        }
    }
    //to simplify the learning, we use the training x set as corpus.
    int index = 0;
    for (std::set<std::string>::iterator it = ptr_x_set_->begin(); it != ptr_x_set_->end(); it++) {
        //std::cout << (*it) << std::endl;
        ptr_x_corpus_map_->insert(std::make_pair((*it),index));
        ptr_x_corpus_->push_back((*it));
        index ++;
    }
    pre_loglikelihood_ = 0;
    start_training_ = false;
}

bool EM::IsIteration() {
    //cal log
    double log_likelihood = 0;
    ptr_Z->clear();
    for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin(); it != ptr_training_seq_->end(); it++) {
        //std::cout << (*it).size()<<std::endl;
        if((*it).size()>0){
            double Z = CalcPX((*it));
            ptr_Z->push_back(Z);
            log_likelihood += log(Z);
        }
    }
    /**
     * below code is for observation only
     */
    std::vector<double> z;
    for(int i=0; i<num_of_training_setence_; i++){
        z.push_back((*ptr_Z)[i]);
    }
    //above code is for observation only.
    std::cout << "the log-likelihood is: "<< log_likelihood <<std::endl;
    double  convergence = log_likelihood - pre_loglikelihood_;
    if(convergence < 0 && start_training_ == true){
        std::cout << "caution: the log-likelihood descreases!!!" <<std::endl;
    }
    if(!start_training_){
        start_training_ = true;
    }
    if(std::abs(convergence) <= EM_ITERATION_STOP){
        std::cout << "iteration stop and training is completed"<<std::endl;
        return false;
    }
    pre_loglikelihood_ = log_likelihood;
    return true;
}

//calc P(X|theta)
double EM::CalcPX(std::vector<std::string> seq) {
    double result = 0;
    int size = seq.size();
    if(1 == size){
        int index = ptr_x_corpus_map_->find(seq[0])->second;
        // P(x_1) = \sum_u P(x_1, y_1 =u) = \sum_u P(x_1|y1_ = u) * p(y_1 = u)
        for(int u=1; u<= HMM_Parameters_.num_of_state_-2; u++){
            result += (*HMM_Parameters_.ptr_e_)[u][index] * (*HMM_Parameters_.ptr_t_)[0][u-1];
        }
        return result;
    }
    double alpha_u_n = 0;
    double beta_u_n = 0;
    for (int u = 1; u < HMM_Parameters_.num_of_state_-2; ++u) {
        alpha_u_n = ptr_fwbw->ForwardResult(ptr_x_corpus_map_,seq,std::make_pair(u,size),HMM_Parameters_);
        beta_u_n = ptr_fwbw->BackwardResult(ptr_x_corpus_map_,seq,std::make_pair(u,size),HMM_Parameters_);
        result += alpha_u_n * beta_u_n;
        if(result ==0){
            std::cout << "ptr_Z is zero"<<std::endl;
        }
    }
    if(result >=1){
        result = 1;
    }
    if(result ==0){
        //std::cout << "ptr_Z is zero"<<std::endl;
    }
    return result;
}

//calc expected count in the entire dataset.
double EM::CalcCountOfUV(std::pair<int, int> uv) {
    std::vector<std::vector<double >> *ptr_t = HMM_Parameters_.ptr_t_;
    std::vector<std::vector<double >> *ptr_e = HMM_Parameters_.ptr_e_;
    int u = uv.first;
    int v = uv.second;
    double count_uv = 0;
    double prob = (double)1 / (double)(HMM_Parameters_.num_of_state_-2);
    int Z_index = 0;
    for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin(); it != ptr_training_seq_->end(); it++) {
        int size = (*it).size();
        if((*it).size() <=0){
            continue;
        }
        double count_i_uv = 0;
        if(1 == size){
            count_i_uv = 0;
            //COUNT(Start, v) = p(y_0 = START, y_1= v|X^i) = p(y_1 = v|X^i) = \frac{a_{START,v} \beta_v(1)} {Z^i};
            int index = ptr_x_corpus_map_->find((*it)[0])->second;
            double numerator =  (*HMM_Parameters_.ptr_t_)[0][v-1] * (*HMM_Parameters_.ptr_e_)[u][index];
            double denominator = (*ptr_Z)[Z_index];
            count_i_uv = numerator / denominator;

        }else{
            for (int j = 1; j <= size-1; j++) {
                int index = ptr_x_corpus_map_->find((*it)[j-1])->second;
                double numerator = 0;
                double alpha_u_j = ptr_fwbw->ForwardResult(ptr_x_corpus_map_,(*it),std::make_pair(u,j),HMM_Parameters_);
                double beta_v_j = ptr_fwbw->BackwardResult(ptr_x_corpus_map_,(*it),std::make_pair(v,j+1),HMM_Parameters_);
                numerator = alpha_u_j * (*ptr_t)[u][v-1] * (*ptr_e)[u][index] * beta_v_j;
                count_i_uv += numerator;
            }
            double denominator = (*ptr_Z)[Z_index];
            if(denominator>0){
                count_i_uv = count_i_uv / denominator;
            }
        }
        Z_index++;
        count_uv += count_i_uv;
    }
    return count_uv;
}

double EM::CalcCountOfU(int u) {
    double count_u = 0;
    int Z_index = 0;
    for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin();
         it != ptr_training_seq_->end(); it++) {
        int seq_size = (*it).size();
        if((*it).size() <=0){
            continue;
        }
        double count_j_u = 0;
        if(1 == seq_size){
            //count(u) = p(y_1 = u|X_i).
            int index = ptr_x_corpus_map_->find((*it)[0])->second;
            double numerator =  (*HMM_Parameters_.ptr_t_)[0][u-1] * (*HMM_Parameters_.ptr_e_)[u][index];
            double denominator = (*ptr_Z)[Z_index];
            count_j_u = numerator / denominator;
        }else{
            for(int j=1; j <= seq_size; j++){
                double alpha = ptr_fwbw->ForwardResult(ptr_x_corpus_map_,(*it),std::make_pair(u,j),HMM_Parameters_);
                double beta = ptr_fwbw->BackwardResult(ptr_x_corpus_map_,(*it),std::make_pair(u,j),HMM_Parameters_);
                double numerator = alpha * beta;
                count_j_u += numerator;
            }
            double denominator = (*ptr_Z)[Z_index];
            if(denominator>0){
                count_j_u = count_j_u / denominator;
            }else{
                std::cout << "Z as denominator is zero"<<std::endl;
            }
        }
        Z_index ++;
        count_u += count_j_u;
    }
    return  count_u;
}

double EM::CalcCountOfUK(std::pair<int, int> uk) {
    int u = uk.first;
    int vk = uk.second;
    double count_uk = 0;
    int Z_index = 0;
    for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin(); it != ptr_training_seq_->end(); it++) {
        std::vector<std::string> str = *it;
        if((*it).size() <=0){
            continue;
        }
        double count_u_j = 0;
        int seq_size = (*it).size();
        double alpha = 0;
        double beta = 0;
        double denominator = (*ptr_Z)[Z_index];
        if(1 == seq_size){
            //COUNT(U -> 0)
            if(str[0] == (*ptr_x_corpus_)[vk]){
                int index = ptr_x_corpus_map_->find((*it)[0])->second;
                double numerator =  (*HMM_Parameters_.ptr_t_)[0][u-1] * (*HMM_Parameters_.ptr_e_)[u][index];
                if(denominator>0){
                    count_u_j = numerator / denominator;
                }
            }
        }else{
            for(int j=1; j<= seq_size; j++) {
                if(str[j-1] == (*ptr_x_corpus_)[vk]){
                    alpha = ptr_fwbw->ForwardResult(ptr_x_corpus_map_,(*it),std::make_pair(u,j),HMM_Parameters_);
                    beta =  ptr_fwbw->BackwardResult(ptr_x_corpus_map_,(*it),std::make_pair(u,j),HMM_Parameters_);
                    double numerator = alpha * beta;
                    count_u_j += numerator;
                    if(count_u_j ==0){
                        std::cout << "cout_u_j equals zero"<<std::endl;
                    }
                }
            }
            if(denominator>0){
                count_u_j = count_u_j / denominator;
            }
        }
        Z_index ++;
        count_uk += count_u_j;
    }
    if(count_uk == 0){
        std::cout << (*ptr_x_corpus_)[vk] << std::endl;
        std::cout << "count_u_k is zero" <<std::endl;
    }
    return count_uk;
}

//  std::cout << "*pptr_alpha_" << seq_no << "," << t << "," << i << "=" << (*pptr_alpha_[seq_no])[t][i] << std::endl;
void EM::EStep() {
    double start, end, cost;
    HMM_Parameters_.ptr_count_u_->clear();
    for(int u=0; u<=HMM_Parameters_.num_of_state_-2; u++){
        //std::cout <<"E-step: u=: "<<u<<std::endl;
        start = clock();
        if(u == 0){
            //calc expected count of COUNT(START, v), here u == START
            for(int v=1; v<=HMM_Parameters_.num_of_state_-2; v++) {
                double Z_index=0;
                double count_START_v = 0;
                for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin(); it != ptr_training_seq_->end(); it++) {
                    if((*it).size() <=0){
                        continue;
                    }
                    double a_START_v = (*HMM_Parameters_.ptr_t_)[0][v-1];
                    double beta_v_1 = ptr_fwbw->BackwardResult(ptr_x_corpus_map_,(*it),std::make_pair(v,1),HMM_Parameters_);
                    double numerator = a_START_v * beta_v_1;
                    double denominator = (*ptr_Z)[Z_index];
                    if(numerator > denominator){
                        std::cout <<"numerator is larger than denominator"<<std::endl;
                    }
                    if(denominator == 0){
                        std::cout << "ptr_Z is zero"<<std::endl;
                    }
                    if(denominator>0){
                        count_START_v += (numerator / denominator);
                    }
                    Z_index ++;
                }
                (*HMM_Parameters_.ptr_count_uv_)[u][v-1] = count_START_v;
            }
            // the expected count of P(y_o = START) == 1, hence, the count for the entire dataset is num_of_training_setence_.
            (*HMM_Parameters_.ptr_count_uv_)[u][HMM_Parameters_.num_of_state_-2] = 0;
            HMM_Parameters_.ptr_count_u_->push_back(num_of_training_setence_);
        }else{
            for(int v=1; v<=HMM_Parameters_.num_of_state_-1; v++){
                if(v==HMM_Parameters_.num_of_state_-1){
                    //calc expected count of COUNT(u, STOP), here v == STOP
                    double Z_index=0;
                    double count_u_STOP = 0;
                    for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin(); it != ptr_training_seq_->end(); it++) {
                        if((*it).size() <=0){
                            continue;
                        }
                        double alpha_u_n = ptr_fwbw->ForwardResult(ptr_x_corpus_map_,(*it),std::make_pair(u,(*it).size()),HMM_Parameters_);
                       double beta_j_n = ptr_fwbw->BackwardResult(ptr_x_corpus_map_,(*it),std::make_pair(u,(*it).size()),HMM_Parameters_);
                       double numerator = alpha_u_n * beta_j_n;
                       double denominator = (*ptr_Z)[Z_index];
                       if(denominator>0){
                           count_u_STOP += numerator / denominator;
                       }
                       Z_index ++;
                    }
                    (*HMM_Parameters_.ptr_count_uv_)[u][v-1] = count_u_STOP;
                }else{
                    (*HMM_Parameters_.ptr_count_uv_)[u][v-1] = CalcCountOfUV(std::make_pair(u,v));
                }
            }
            HMM_Parameters_.ptr_count_u_->push_back(CalcCountOfU(u));
        }
        end = clock();
        cost = end - start;
        //std::cout << "CPU cycle for expected count UV is:" <<cost << std::endl;
        start = clock();
        //calc expected count of b_u_k
        if(u>=1){
            for(int k=0; k< number_of_x_; k++){
                (*HMM_Parameters_.ptr_count_uk_)[u][k] = CalcCountOfUK(std::make_pair(u,k));
            }
        }
        end = clock();
        cost = end - start;
        //std::cout << "CPU cycle for expected count UK is:" <<cost << std::endl;

    }
}

void EM::MStep() {
    std::cout << "M-Step start" << std::endl;
    for(int u=0; u<=HMM_Parameters_.num_of_state_-2; u++){
        double t = 0;
        for(int v=1; v<=HMM_Parameters_.num_of_state_-1; v++){
            double a = (*HMM_Parameters_.ptr_count_uv_)[u][v-1];
            double b =  (*HMM_Parameters_.ptr_count_u_)[u];
            (*HMM_Parameters_.ptr_t_)[u][v-1] = (*HMM_Parameters_.ptr_count_uv_)[u][v-1] / (*HMM_Parameters_.ptr_count_u_)[u];
            //std::cout << "<u, v>: <"<<u<<","<<v-1<<">: "<<(*HMM_Parameters_.ptr_t_next_)[u][v-1]<<std::endl;
            t+=(*HMM_Parameters_.ptr_t_)[u][v-1];
        }
        //std::cout << "the sum of t for "<< u <<"th row is:"<<t<<std::endl;
        if(u>=1){
            double e =0;
            for(int k=0; k<number_of_x_; k++){
                double a = (*HMM_Parameters_.ptr_count_uk_)[u][k];
                double b = (*HMM_Parameters_.ptr_count_u_)[u];
                (*HMM_Parameters_.ptr_e_)[u][k] =  (*HMM_Parameters_.ptr_count_uk_)[u][k] / (*HMM_Parameters_.ptr_count_u_)[u];
                if(0 == (*HMM_Parameters_.ptr_e_)[u][k]){
                    //std::cout << "the value is zero" <<std::endl;
                }
                e+=(*HMM_Parameters_.ptr_e_)[u][k];
                //std::cout << "<u, k>: <"<<u<<","<<k<<">: "<<(*HMM_Parameters_.ptr_e_next_)[u][k]<<std::endl;
            }
            //std::cout << "the sum of e for "<< u <<"th row is:"<<e<<std::endl;

        }
     }
}

void EM::UpdateParameters() {
    for(int u=0; u<=HMM_Parameters_.num_of_state_-2; u++){
        for(int v=1; v<HMM_Parameters_.num_of_state_-1; v++){
            (*HMM_Parameters_.ptr_t_)[u][v-1] = (*HMM_Parameters_.ptr_t_next_)[u][v-1] ;
        }
        if(u>=1){
            for(int k=0; k<number_of_x_; k++){
                (*HMM_Parameters_.ptr_e_)[u][k] = (*HMM_Parameters_.ptr_e_next_)[u][k];
            }
        }
    }
    //observation only
    for(int u=0; u<=HMM_Parameters_.num_of_state_-2; u++){
        double uv = 0;
        double uk = 0;
        for(int v=1; v<=HMM_Parameters_.num_of_state_-1; v++){
           uv += (*HMM_Parameters_.ptr_t_)[u][v-1];
        }
        //std::cout<<"trans prob of the "<<u<<"th row is: "<<uv<<std::endl;
        if(u>=1){
            for(int k=0; k<number_of_x_; k++){
               double prob =  (*HMM_Parameters_.ptr_e_)[u][k];
               uk += prob;
            }
            //std::cout<<"emission prob of the "<<u<<"th row is: "<<uk<<std::endl;
        }
    }
}

void EM::Learning() {
    Init();
    int num_of_interation = 0;
    while(IsIteration()){
        std::cout<<"this is the "<<num_of_interation<<" th interation"<<std::endl;
        EStep();
        MStep();
       // UpdateParameters();
        num_of_interation++;
    }
}

void EM::GenerateSeqFromVector(std::vector<std::string> *ptr_vector,
                                     std::vector<std::vector<std::string>> *ptr_seq_vector) {
    std::vector<std::string> seq;
    for (std::vector<std::string>::iterator it = ptr_vector->begin(); it != ptr_vector->end(); it++) {
//    std::cout<<*it<<std::endl;
        if (*it == SPERATOR_FLAG) {
            ptr_seq_vector->push_back(seq);
            seq.clear();
            continue;
        } else {
            seq.push_back(*it);
            // do not forget the last seq which doesn't contain a SPEARATOR_FLAG at the end.
            if (it == (ptr_vector->end() - 1)) {
                ptr_seq_vector->push_back(seq);
            }
        }
    }
}