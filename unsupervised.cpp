//
// Created by  ngs on 05/05/2018.
//
#include <numeric>
#include "unsupervised.h"

Learning::Learning(const char *file_name, DatasetMgr *ptr_datamgr) {
    ptr_datamgr_ = ptr_datamgr;
    //state num
    ptr_state_set_ = ptr_datamgr_->GetTagSet();
    number_of_state_ = ptr_datamgr_->GetTagSet()->size();
    //training x number
    ptr_x_set_ = ptr_datamgr->GetTrainingXSet();
    number_of_x_ = ptr_datamgr->GetTrainingXSet()->size();
    ptr_train_x_vector_ = ptr_datamgr->GetTrainingXVector();
    num_of_training_setence_ = ptr_datamgr->GetNumOfTrainingSeqs();
    ptr_training_seq_ = new std::vector<std::vector<std::string>>();
    GenerateSeqFromVector(ptr_train_x_vector_, ptr_training_seq_);
    //pi
    ptr_pi_ = new std::vector<double>(number_of_state_);
    //transition maxtrix
    ptr_a_ = new std::vector<std::vector<double>>(number_of_state_, std::vector<double>(number_of_state_, 1));
    //emission matrix
    ptr_b_ = new std::vector<std::vector<double>>(number_of_state_, std::vector<double>(number_of_x_, 1));
    //pi
    ptr_next_pi_ = new std::vector<double>(number_of_state_);
    //transition maxtrix
    ptr_next_a_ = new std::vector<std::vector<double>>(number_of_state_, std::vector<double>(number_of_state_, 1));
    //emission matrix
    ptr_next_b_ = new std::vector<std::vector<double>>(number_of_state_, std::vector<double>(number_of_x_, 1));
    pptr_alpha_ = new std::vector<std::vector<double>> *[num_of_training_setence_];
    pptr_beta_ = new std::vector<std::vector<double>> *[num_of_training_setence_];
    int seq_no = 0;
    for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin();
         it != ptr_training_seq_->end(); it++) {
        int size = (*it).size();
        //std::cout << "sequence is: ";
        for (std::vector<std::string>::iterator itt = (*it).begin(); itt != (*it).end(); itt++) {
            //std::cout << *itt << " ";
        }
        //std::cout << std::endl;
        pptr_alpha_[seq_no] = new std::vector<std::vector<double>>(size, std::vector<double>(number_of_state_, 1));
        pptr_beta_[seq_no] = new std::vector<std::vector<double>>(size, std::vector<double>(number_of_state_, 1));
        seq_no++;
    }
    ptr_PO_ = new std::vector<double>;
    ptr_x_corpus_ = new std::vector<std::string>;
    pre_loglikelihood_ = INITIAL_LOG_LIKEIHOOD;
    current_loglikelihood_ = 0;
}

Learning::~Learning() {
    delete ptr_a_;
    delete ptr_pi_;
    delete ptr_b_;
    delete ptr_next_a_;
    delete ptr_next_pi_;
    delete ptr_next_b_;
    int seq_no = 0;
    for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin();
         it != ptr_training_seq_->end(); it++) {
        delete pptr_alpha_[seq_no];
        delete pptr_beta_[seq_no];
        seq_no++;
    }
    delete pptr_alpha_;
    delete pptr_beta_;
    delete ptr_training_seq_;
    delete ptr_PO_;
    delete ptr_x_corpus_;
}

void Learning::Init() {
    // we simply set state number as the one in tagged sequences. A more practical solution is expected to be developed.
    for (int i = 0; i < number_of_state_; i++) {
        //init pi
        (*ptr_pi_)[i] = (double) 1 / (double) number_of_state_;
        (*ptr_next_pi_)[i] = 0;
        //std::cout << "ptr_pi_" <<i<<"="<<(*ptr_pi_)[i]<<std::endl;
        //init transition prob
        for (int j = 0; j < number_of_state_; j++) {
            (*ptr_a_)[i][j] = (double) 1 / (double) number_of_state_;
            (*ptr_next_a_)[i][j] = 0;
            //std::cout << "ptr_a_" <<i<<","<<j<<"="<<(*ptr_a_)[i][j]<<std::endl;
        }
        //init emission prob
        for (int k = 0; k < number_of_x_; k++) {
            (*ptr_b_)[i][k] = (double) 1 / (double) number_of_x_;
            (*ptr_next_b_)[i][k] = 0;
            // std::cout << "ptr_b_" <<i<<","<<k<<"="<<(*ptr_b_)[i][k]<<std::endl;
        }
    }
    // init forward & backward matrixs of all sequences.
    int seq_no = 0;
    for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin();
         it != ptr_training_seq_->end(); it++) {
//        int size = (*it).size();
        for (int t = 0; t < (*it).size(); t++) {
            for (int i = 0; i < number_of_state_; i++) {
                (*pptr_alpha_[seq_no])[t][i] = 0.001;
                (*pptr_beta_[seq_no])[t][i] = 0.001;
                //  std::cout << "*pptr_alpha_" << seq_no << "," << t << "," << i << "=" << (*pptr_alpha_[seq_no])[t][i] << std::endl;
            }
        }
        seq_no++;
    }
    //to simplify the learning, we use the training x set as corpus.
    for (std::set<std::string>::iterator it = ptr_x_set_->begin(); it != ptr_x_set_->end(); it++) {
        ptr_x_corpus_->push_back(*it);
    }
}

/**
 *
 * @param sequence
 * @param index
 * @return
 */

std::pair<double, double> Learning::Forward(const std::vector<std::string> sequence, size_t seq_no) {
    double alpha_t_i = 0; //p_index indicates P(o_1, o_2...o_index, i_index=q_i|lamda)
    double p = 0; // p indicates P(O|lamda)
    for (int i = 0; i < number_of_state_; i++) {
        //init;
        (*pptr_alpha_[seq_no])[0][i] = (*ptr_pi_)[i] * (*ptr_b_)[i][0];
        double alpha = 0;
        for (int t = 0; t < sequence.size() - 1; t++) {
            // recurrent
            for (int j = 0; j < number_of_state_; j++) {
                alpha += (*pptr_alpha_[seq_no])[t][j] * (*ptr_a_)[j][i];
            }
            (*pptr_alpha_[seq_no])[t + 1][i] = alpha * (*ptr_b_)[i][t + 1];
        }
        //P(O|lemda) = sum_i alpha[T-1][i]
        p += (*pptr_alpha_[seq_no])[sequence.size() - 1][i];
    }
    //calc log likelihood
    double likelh = log(p);
    //return
    return std::make_pair(p, likelh);
}

void Learning::Backward(const std::vector<std::string> sequence, size_t seq_no) {
    double beta_t_i = 0;
    for (int i = 0; i < number_of_state_; ++i) {
        //initialization
        double beta = 0;
        (*pptr_beta_[seq_no])[sequence.size() - 1][i] = 1;
        for (int t = sequence.size() - 2; t >= 0; t--) {
            for (int j = 0; j < number_of_state_; ++j) {
                beta += (*ptr_a_)[i][j] * (*ptr_b_)[j][t + 1] * (*pptr_beta_[seq_no])[t + 1][j];
            }
            (*pptr_beta_[seq_no])[t][i] = beta;
        }
    }
}

void Learning::GenerateSeqFromVector(std::vector<std::string> *ptr_vector,
                                     std::vector<std::vector<std::string>> *ptr_seq_vector) {
    std::vector<std::string> seq;
    for (std::vector<std::string>::iterator it = ptr_vector->begin() + 1; it != ptr_vector->end(); it++) {
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

double Learning::CalcGamma(size_t seq_no, size_t t, size_t i) {
    double gamma = 0;
    gamma = (*pptr_alpha_[seq_no])[t][i] * (*pptr_beta_[seq_no])[t][i];
    gamma = gamma / (*ptr_PO_)[seq_no];
    return gamma;
}

double Learning::CalcXi(size_t seq_no, size_t t, size_t i, size_t j) {
    double xi = 0;
    xi = (*pptr_alpha_[seq_no])[t][i] * (*ptr_a_)[i][j] * (*ptr_b_)[j][t + 1] * (*pptr_beta_[seq_no])[t + 1][j];
    xi = xi / (*ptr_PO_)[seq_no];
    return xi;
}

void Learning::CalcPi() {
    for (int i = 0; i < number_of_state_; i++) {
        double pi = 0;
        for (int seq_no = 0; seq_no < num_of_training_setence_; seq_no++) {
            pi += CalcGamma(seq_no, 0, i);
        }
        (*ptr_next_pi_)[i] = pi;
    }
    //normalization;
    double sum = std::accumulate(ptr_next_pi_->begin(), ptr_next_pi_->end(), 0);
    for (int i = 0; i < number_of_state_; i++) {
        double pi = (*ptr_next_pi_)[i] / sum;
        (*ptr_next_pi_)[i] = pi;
    }
}

void Learning::CalcA() {
    //calc a_ij
    for (int i = 0; i < number_of_state_; i++) {
        double a_ij = 0;
        double xi_tij = 0;
        double gamma_ti = 0;
        for (int j = 0; j < number_of_state_; j++) {
            a_ij = 0;
            for (int seq_no = 0; seq_no < num_of_training_setence_; seq_no++) {
                xi_tij = 0;
                gamma_ti = 0;
                for (int t = 0; t < (*ptr_training_seq_)[seq_no].size() - 1; t++) {
                    xi_tij += CalcXi(seq_no, t, i, j);
                    gamma_ti += CalcGamma(seq_no, t, i);
                }
                double a = 0;
                if (gamma_ti > 0) {
                    a = xi_tij / gamma_ti;
                }
                a_ij += a;
            }
            (*ptr_next_a_)[i][j] = a_ij;
        }
    }
    //normalization
    for (int i = 0; i < number_of_state_; i++) {
        double sum = std::accumulate((*ptr_next_a_)[i].begin(), (*ptr_next_a_)[i].end(), 0);
        for (int j = 0; j < number_of_state_; j++) {
//      std::cout << "before is "<< (*ptr_next_a_)[i][j]<<std::endl;
            double a = (*ptr_next_a_)[i][j];
            (*ptr_next_a_)[i][j] = a / sum;
//      std::cout << "after is "<< (*ptr_next_a_)[i][j]<<std::endl;
        }
    }
}

void Learning::CalcB() {
    //calc b_jk
    for (int j = 0; j < number_of_state_; j++) {
        double b_jk = 0;
        double gamma_tj = 0;
        double gamma_tjk = 0;
        for (int k = 0; k < number_of_x_; k++) {
            b_jk = 0;
            std::string v_k = (*ptr_x_corpus_)[k];
            for (int seq_no = 0; seq_no < num_of_training_setence_; seq_no++) {
                gamma_tj = 0;
                gamma_tjk = 0;
                for (int t = 0; t < (*ptr_training_seq_)[seq_no].size(); t++) {
                    double b = CalcGamma(seq_no, t, j);
                    std::string o_t = (*ptr_training_seq_)[seq_no][t];
                    if (0 == o_t.compare(v_k)) {
                        gamma_tjk += b;
                    }
                    gamma_tj += b;
                }
                double bb = 0;
                if (gamma_tj > 0) {
                    bb = gamma_tjk / gamma_tj;
                }
                //accumulation for each sequence.
                b_jk += bb;
            }
            (*ptr_next_b_)[j][k] = b_jk;
        }
    }
    //normalization
    for (int j = 0; j < number_of_state_; ++j) {
        double sum = std::accumulate((*ptr_next_b_)[j].begin(), (*ptr_next_b_)[j].end(),0);
        for(int k=0; k< number_of_x_; ++k){
            (*ptr_next_b_)[j][k] /= sum;
        }
    }
}


bool Learning::IsIteration(size_t iteration_no) {
    int seq_no = 0;
    ptr_PO_->clear();
    //perform forward and backward, and accumulation of log-likelihood for all sequences.
    for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin();
         it != ptr_training_seq_->end(); it++) {
        current_loglikelihood_ = 0;
        std::pair<double, double> x = Forward(*it, seq_no);
        Backward(*it, seq_no);
        ptr_PO_->push_back(x.first);
        current_loglikelihood_ += x.second;
        seq_no++;
    }
    //log likelihood
    if (current_loglikelihood_ > pre_loglikelihood_) {
        //it means that log-likelihood has converged.
        if (current_loglikelihood_ - pre_loglikelihood_ <= EM_ITERATION_STOP) {
            return false;
        }
    } else {
        std::cout << "log-likelihood decreases at the " << iteration_no << " th iteration" << std::endl;
    }
    std::cout << iteration_no << " th iterations: log-likelihood is:" << current_loglikelihood_ << std::endl;
    pre_loglikelihood_ = current_loglikelihood_;
    return true;
}

void Learning::UpdateParameters() {
    for (int i = 0; i < number_of_state_; i++) {
        for (int j = 0; j < number_of_state_; ++j) {
            (*ptr_a_)[i][j] = (*ptr_next_a_)[i][j];
        }
        (*ptr_pi_)[i] = (*ptr_next_pi_)[i];
        for (int k = 0; k < number_of_x_; k++) {
            (*ptr_b_)[i][k] = (*ptr_next_b_)[i][k];
        }
    }
    //ptr_next_a_->clear();
    //ptr_next_b_->clear();
    //ptr_next_pi_->clear();
    ptr_PO_->clear();
    for (int i = 0; i < num_of_training_setence_; i++) {
        //pptr_alpha_[i]->clear();
        //pptr_beta_[i]->clear();
    }
}

void Learning::BaumWelch() {
    int iteration_no = 0;
    while (true) {
        //E-step;
        if (!IsIteration(iteration_no)) {
            break; // quit the EM algorithm when log-likelihood has converged.
        }
        //M-step;
        CalcPi();
        CalcA();
        CalcB();
        UpdateParameters();
        iteration_no++;
    }
}

/**
void Learning::BaumWelch() {
  //calc
  bool isiteration = true;
  double maxloglh = 0;
  int seq_no = 0;

  std::vector<std::vector<int>> alpha;
  double loglikelihd = 0; //
  std::vector<double> prob_o;
  GenerateSeqFromVector(ptr_train_x_vector_, ptr_training_seq_);
  while (isiteration) {
    prob_o.clear();
    loglikelihd = 0;
    // perform forward and backward, loglike
    for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin();
         it != ptr_training_seq_->end(); it++) {
      const std::vector<std::string> seq = *it;
      std::pair<double, double> x = Forward(seq, seq_no);
      Backward(seq, seq_no);
      prob_o.push_back(x.first);
      loglikelihd += x.second;
      seq_no++;
    }

    //calc pi_i
    double pi_i = 0;
    for (int i = 0; i < number_of_state_; i++) {
      for (seq_no = 0; seq_no < num_of_training_setence_; seq_no++) {
        double p_o_i_1 = (*pptr_alpha_)[seq_no][0][i] * (*pptr_beta_)[seq_no][0][i];
        pi_i += p_o_i_1 / prob_o[seq_no];
      }
      (*ptr_pi_)[i] = pi_i;
    }

    //calc a_{i,j}
    double a_i_j = 0;
    double p_o_i_t = 0;
    double p_o_i_j_t = 0;
    for (int i = 0; i < number_of_state_; i++) {
      for (int j = 0; j < number_of_state_; ++j) {
        for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin();
             it != ptr_training_seq_->end(); it++) {
          for (int t = 0; t < (*it).size(); t++) {
            p_o_i_j_t += (*pptr_alpha_)[seq_no][t][i] * (*ptr_a_)[i][j] * (*ptr_b_)[t + 1][j] *
                         (*pptr_beta_)[seq_no][t + 1][i];
            p_o_i_t += (*pptr_alpha_)[seq_no][t][i] * (*pptr_beta_)[seq_no][t][i];
          }
          a_i_j += (p_o_i_j_t / p_o_i_t);
        }
        (*ptr_b_)[i][j] = a_i_j;
      }
    }

    //calc b_{j,k}
    //double b_j_k = 0;
    double p_o_t_j_v = 0;
    double p_o_t_j = 0;
    seq_no = 0;
    std::vector<std::vector<double>> b_j_k;
    for (int j = 0; j < number_of_state_; ++j) {
        int k=0;
        bool is_k_exceed = false;
        while(!is_k_exceed) {
          for (std::vector<std::vector<std::string>>::iterator it = ptr_training_seq_->begin();
               it != ptr_training_seq_->end(); it++) {
              if( k < (*it).size()) {
                k++;
              } else {
                is_k_exceed = true;
              }
              for (int t = 0; t < (*it).size(); t++) {
              p_o_t_j += (*pptr_alpha_)[seq_no][t][j] * (*pptr_beta_)[seq_no][t][j];
              if ((*it)[t] == (*it)[k]) {
                p_o_t_j_v += (*pptr_alpha_)[seq_no][t][j] * (*pptr_beta_)[seq_no][t][j];
              }
            }
            //b_j_k.push_back(p_o_t_j_v / p_o_t_j);
          }
        }
      }
    }
  }
}
**/

void Learning::StartTraining() {
    Init();
    BaumWelch();
}