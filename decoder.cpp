//
// Created by  ngs on 28/04/2018.
//

#include "decoder.h"
#include <cctype>
Decoder::Decoder(const char * test_file_name,DatasetMgr *ptr_datasetmgr) {
    ptr_datasetmgr_  = ptr_datasetmgr;
    ptr_datasetmgr_->OpenDataSet(test_file_name, false);
    ptr_x_to_state_docoding_ =  new std::map<std::string, std::string>();
    ptr_path_str_vector_ = new std::vector<std::string>();
    ptr_path_prob_vector_ = new std::vector<std::string>();
    ptr_state_path_ = new std::map<std::string,Node>();
}

Decoder::~Decoder() {
    delete ptr_x_to_state_docoding_;
    delete ptr_path_str_vector_;
    delete ptr_path_prob_vector_;
    delete ptr_state_path_;
}

void Decoder::Decoding() {
    Init();
    std::vector<std::string> setence;
    std::vector<std::string>::iterator next_it;
    for (std::vector<std::string>::iterator it = (ptr_test_x_vector_->begin() + 1);it != ptr_test_x_vector_->end(); ++it) {
        if ((it + 1) != ptr_test_x_vector_->cend()) {
            next_it = it + 1;
        } else{
            continue;
        }

        if (*it != SPERATOR_FLAG) {
            setence.push_back(*it);
        } else{
            continue;
        }

        if (*next_it == SPERATOR_FLAG) {
            if (1 == setence.size()) {
                std::string state = SearchStateOfMaxProb(*it);
                ptr_x_to_state_docoding_->insert(make_pair(*it, state));
                setence.clear();
            } else {
                ptr_state_path_->clear();
                Vertibi(setence);
                Output(setence);
                setence.clear();
            }
        }
    }
    for (std::map<std::string, std::string>::iterator it = ptr_x_to_state_docoding_->begin();
         it != ptr_x_to_state_docoding_->end(); ++it) {
         //std::cout << "decoding result: x= "<<it->first<<"; y="<<it->second <<std::endl;
    }
}

void Decoder::Init() {
    ptr_tag_set_ = ptr_datasetmgr_->GetTagSet();
    ptr_test_x_vector_ = ptr_datasetmgr_->GetTestFeatureVector();
    ptr_test_tag_vector_ = ptr_datasetmgr_->GetTestFlagVector();
    ptr_state_to_state_prob_map_ = ptr_datasetmgr_->GetStateTransProbMap();
    ptr_state_to_x_prob_map_ = ptr_datasetmgr_->GetEmissionProbMap();
}

std::string &Decoder::SearchStateOfMaxProb(std::string str_x) {
    double prob = 0;
    double max_prob = 0;
    std::string max_state = NOT_FOUND_FLAG;
    std::map<std::string, double >::iterator indictor;
    for(std::set<std::string>::iterator it = ptr_tag_set_->begin();it!=ptr_tag_set_->end();++it){
        std::string state = *it;
        state+=SPERATOR_FLAG;
        if(isdigit(*(str_x.c_str()))){
            str_x = std::to_string(DIGITAL_FLAG);
        }
        state+=str_x;
        //find the largest prob in the prob map;
        indictor = ptr_state_to_x_prob_map_->find(state);
        if(indictor != ptr_state_to_x_prob_map_->end()){
            prob = indictor->second;
            if(prob > max_prob){
                max_prob = prob;
                max_state = *it;
            }
        }
    }
    return max_state;

}

void Decoder::Vertibi(std::vector<std::string> observation) {
    for(std::vector<std::string>::iterator it = observation.begin(); it != observation.end(); ++it){
        for(std::set<std::string>::iterator itt = ptr_tag_set_->begin();itt!=ptr_tag_set_->end();++itt){
            double emission_prob = GetEmissionProb(*ptr_state_to_x_prob_map_,*it,*itt);
            if(emission_prob == 0){
                emission_prob = SMOOTH_VALUE;
            }
            Node max_node;
            max_node.state = *itt;
            if(it == observation.begin()) {
                max_node.pre_state = PATH_START;
                max_node.path = PATH_START;
                max_node.prob = emission_prob;
                ptr_state_path_->insert(make_pair(*itt,max_node));
            }else {
                double max_prob = 0;
                for(std::set<std::string>::iterator ittt = ptr_tag_set_->begin();ittt!=ptr_tag_set_->end();++ittt) {
                    std::string state_to_state = *ittt + SPERATOR_FLAG+ *itt;
                    Node node;
                    GetNode(*ittt,node);
                    double state_trans_prob = FindProb(*ptr_state_to_state_prob_map_, state_to_state);
                    if(state_trans_prob == 0){
                        state_trans_prob = SMOOTH_VALUE;
                    }
                    double decoding_prob = node.prob * state_trans_prob * emission_prob;
                    if (decoding_prob > max_prob) {
                        max_prob = decoding_prob;
                        max_node.pre_state = *ittt;
                        if(it == (observation.end()-1)){
                            max_node.path = node.path + SPERATOR_FLAG + *ittt + SPERATOR_FLAG + *itt;
                        }else{
                            max_node.path = node.path + SPERATOR_FLAG + *ittt;
                        }
                        max_node.prob = decoding_prob;
                    }
                }
                ResetNode(max_node.state,max_node);
            }
        }
    }
}

double Decoder::FindProb(std::map<std::string, double> &prob_map, std::string key_str) {
    std::map<std::string, double>::iterator indictor = prob_map.find(key_str);
    if(indictor != prob_map.end()){
        return indictor->second;
    } else{
        return 0;
    }
}

double Decoder::GetEmissionProb(std::map<std::string, double> &prob_map, std::string observation_str,
                                std::string state_str) {
    std::string state = state_str;
    state+=SPERATOR_FLAG;
    if(isdigit(*(observation_str.c_str()))){
        std::string digit = std::to_string(DIGITAL_FLAG);
        state+=digit;
    }else{
        state+=observation_str;
    }
    return FindProb(prob_map,state);
}

void Decoder::GetNode(std::string state, Node &node) {
    std::map<std::string, Node>::iterator it = ptr_state_path_->find(state);
    if(it!= ptr_state_path_->end()){
        node = it->second;
    }else{
        node.state = NOT_FOUND;
    }
}

void Decoder::ResetNode(std::string state, Node &node) {
    ptr_state_path_->erase(state);
    ptr_state_path_->insert(make_pair(state, node));
}

void Decoder::Output(std::vector<std::string> observation_sentence) {
    Node max_node;
    std::string str;
    double max_prob = 0;
    for(std::vector<std::string>::iterator it = observation_sentence.begin(); it != observation_sentence.end(); ++it) {
        str+=" ";
        str+=*it;
    }
    for(std::map<std::string, Node>::iterator itt = ptr_state_path_->begin();itt!=ptr_state_path_->end();++itt){
        std::string state = itt->first;
        Node node = itt->second;
        //std::cout<<"The sentence "<<str<<", the node path for state "<<state<<" is "<<node.path<<", prob is "<<node.prob<<std::endl;
        if(node.prob > max_prob){
            max_prob = node.prob;
            max_node = node;
        }
    }
    //std::cout<<"The sentence "<<str<<", the max node path for state is "<<max_node.path<<", max prob is "<<max_node.prob<<std::endl;
}
