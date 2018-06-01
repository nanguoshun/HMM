//
// Created by  ngs on 27/04/2018.
//
#include "datasetmgr.h"
#include <cctype>

DatasetMgr::DatasetMgr(bool is_sentence_level){
    is_sentence_level_ = is_sentence_level;
    ptr_line_ = new char[LINE_MAX_SIZE]();
    ptr_tag_set_ = new std::set<std::string>();
    ptr_tag_vector_ = new std::vector<std::string>();
    ptr_pair_tag_vector_ = new std::vector<std::string>();
    ptr_tag_count_map_ = new std::map<std::string, size_t >();
    ptr_pair_tag_count_map_ = new std::map<std::string, size_t >();
    ptr_state_to_state_prob_map_ = new std::map<std::string, double >();
    ptr_x_tag_vector_ = new std::vector<std::string>();
    ptr_x_tag_count_map_ = new std::map<std::string, size_t >();
    ptr_state_to_x_prob_map_ = new std::map<std::string, double >();
    ptr_test_x_vector_  = new std::vector<std::string>();
    ptr_test_tag_vector_ = new std::vector<std::string>();

    ptr_x_vector_ = new std::vector<std::string>();

    ptr_x_set_ = new std::set<std::string>();
    num_of_training_setence_ = 0;
}

DatasetMgr::~DatasetMgr() {
    delete []ptr_line_;
    delete ptr_tag_set_;
    delete ptr_tag_vector_;
    delete ptr_pair_tag_vector_;
    delete ptr_tag_count_map_;
    delete ptr_pair_tag_count_map_;
    delete ptr_state_to_state_prob_map_;
    delete ptr_x_tag_vector_;
    delete ptr_x_tag_count_map_;
    delete ptr_state_to_x_prob_map_;
    delete ptr_test_x_vector_;
    delete ptr_test_tag_vector_;
    delete ptr_x_vector_;
    delete ptr_x_set_;
}

/**
 * Extract features and tags from training set and store them into vectors.
 *
 * @param file_name
 * @return
 */
bool DatasetMgr:: OpenDataSet(const char *file_name, bool istraining) {
    std::ifstream ifs(file_name);
    std::vector<std::string> line_vector;
    size_t i = 0;
    while(ifs.getline(ptr_line_, LINE_MAX_SIZE)){
        if('\0' == ptr_line_[0] || '\t'==ptr_line_[0]||' ' == ptr_line_[0]){
            continue;
        }
        if('.' == ptr_line_[0]){
            if(is_sentence_level_){
                ptr_x_vector_->push_back(SPERATOR_FLAG);
                ptr_tag_vector_->push_back(SPERATOR_FLAG);
                num_of_training_setence_++;
                continue;
            }else{
                continue;
            }
        }
        i++;
        line_vector.clear();
        if(true == Tokenized(ptr_line_,"\t ",&line_vector,TAG_MAX_SIZE,istraining)){
            if(true == istraining){
                OpenTrainSet(&line_vector,is_sentence_level_);
            }else{
                OpenTestSet(&line_vector);
            }
        }
    }
    for(std::vector<std::string>::iterator it = ptr_test_x_vector_->begin();it!=ptr_test_x_vector_->end();++it){
            //std::cout << "the tag in test set is: "<<*it<<std::endl;
    }
    return true;
}

void DatasetMgr::OpenTrainSet(std::vector<std::string> *ptr_vector, bool is_sentence_level) {
    std::vector<std::string>::iterator it_x = ptr_vector->begin();
    std::vector<std::string>::iterator it_tag = ptr_vector->end()-2;
    std::vector<std::string>::iterator it_parse = ptr_vector->end()-1;
    if(is_sentence_level){
     //   std::cout << *it_x << std::endl;
     //   std::cout << *it_tag << std::endl;
        ptr_x_vector_->push_back(*it_x);
        ptr_tag_vector_->push_back(*it_tag);
        ptr_tag_set_->insert(*it_tag);
        ptr_x_set_->insert(*it_x);
    }else{
        ptr_tag_set_->insert(*it_tag);
        //BOI tag, the "O" is indicated as OUT_FLAG, it indicates that no state transition for a "0" sentence.
        if(*it_parse == TAGER_BIO_O){
            ptr_tag_vector_->push_back(OUT_FLAG);
        }
        //transfer the training x into a vector, each sentence is separated by SEPARATOR.
        if(*it_parse == TAGER_BIO_B || *it_parse == TAGER_BIO_O){
            ptr_x_vector_->push_back(SPERATOR_FLAG);
            num_of_training_setence_++;
        }
        ptr_x_vector_->push_back(*it_x);
        //features without duplicate.
        ptr_x_set_->insert(*it_x);
        ptr_tag_vector_->push_back(*it_tag);
        std::string x = *it_tag;
        MergeTwoString(&x,*it_x,SPERATOR_FLAG);
        ptr_x_tag_vector_->push_back(x);
    }
}

void DatasetMgr::OpenTestSet(std::vector<std::string> *ptr_vector) {
    std::vector<std::string>::iterator it_x = ptr_vector->begin();
    std::vector<std::string>::iterator it_tag = ptr_vector->begin()+1;
    std::vector<std::string>::iterator it_parse = ptr_vector->end()-1;
    //to facilitate decoding, we insert a SPERATOR_FLAG for each sentence or for each "O" word
    if(*it_parse == TAGER_BIO_B || *it_parse == TAGER_BIO_O){
        ptr_test_x_vector_->push_back(SPERATOR_FLAG);
        ptr_test_tag_vector_->push_back(SPERATOR_FLAG); // to calculate PR
    }
    ptr_test_x_vector_->push_back(*it_x);
    ptr_test_tag_vector_->push_back(*it_tag);
}

bool DatasetMgr::MergeTwoString(std::string *ptr_str1, std::string str2, std::string separator) {
    if(ptr_str1){
        *ptr_str1 += separator;
        *ptr_str1 += str2;
        return true;
    }
    return false;
}

/**
 * Read a line and extract features and tag
 *
 * @param ptr_line
 * @param ptr_space
 * @param ptr_tagset
 * @param tag_maxsize
 * @return  false: the feature is a punctuation, ture: otherwise
 */
bool DatasetMgr::Tokenized(char *ptr_line, const char *ptr_space, std::vector<std::string> *ptr_string_line, size_t tag_maxsize,bool istraining) {
    char * endofline = ptr_line + std::strlen(ptr_line);
    const char * endofspace = ptr_space + std::strlen(ptr_space);
    size_t  size=0;
    while(size < tag_maxsize){
        char *space = std::find_first_of(ptr_line,endofline,ptr_space,endofspace); //search the space in the line.
        *space = '\0';
        if(*ptr_line!='\0'){
            //if (!ispunct(*ptr_line)){ // omit if it is a punctuation, such as ;, ?
                if(istraining){
                    if(!isdigit(*ptr_line)){
                        ptr_string_line->push_back(ptr_line);
                    }else{ //unify as a FLAG if it is a digit, such as 120, 34.5.
                        ptr_string_line->push_back(std::to_string(DIGITAL_FLAG));
                    }
                }else{
                    ptr_string_line->push_back(ptr_line);
                }
           // }else{
           //     return false;
           // }
            ++size;
        }
        if(space == endofline){
            break;
        }
        ptr_line = space + 1;
    }
    return true;
}
/**
 * Calc the count from training dataset.
 *
 * @param ptr_vector
 * @param ptr_count_map
 * @param option
 */
void DatasetMgr::GenerateCountMap(std::vector<std::string> *ptr_vector, std::map<std::string, size_t> *ptr_count_map, bool option) {
    //calc the count.
    if(option){
        for(std::vector<std::string>::iterator it = ptr_vector->begin();it!=ptr_vector->end();++it) {
            size_t count = std::count(ptr_vector->begin(),ptr_vector->end(),*it);
            ptr_count_map->insert(make_pair(*it,count));
            //std::cout << "The count of "<<*it<<" is "<<count<<std::endl;
        }
    }else{
        for(std::vector<std::string>::iterator it = ptr_vector->begin();it!=ptr_vector->end();++it) {
            if(*it!=OUT_FLAG){
                size_t count = std::count(ptr_vector->begin(),ptr_vector->end(),*it);
                ptr_count_map->insert(make_pair(*it,count));
            }
        }
    }
}

/**
 * Generate the transition vector of two hidden states from training dataset.
 */
void DatasetMgr::GenerateStateTransitionVector() {
    // generate the transition vector.
    for(std::vector<std::string>::iterator it = ptr_tag_vector_->begin();it!=ptr_tag_vector_->end();++it){
        if((it+1) !=ptr_tag_vector_->end()){
            std::vector<std::string>::iterator next_it = it + 1;
            if(*it != OUT_FLAG){
                if(*next_it != OUT_FLAG){
                    std::string pair = *it;
                    MergeTwoString(&pair, *next_it,SPERATOR_FLAG);
                    ptr_pair_tag_vector_->push_back(pair);
                }
            } else{
                it = next_it;
            }
        }
    }
}

void DatasetMgr::CalcProb(std::map<std::string, size_t> *ptr_cout, std::map<std::string, size_t> *ptr_trans_cout,
                          std::map<std::string, double> *ptr_prob) {
    for(std::map<std::string,size_t >::iterator it = ptr_trans_cout->begin();it!=ptr_trans_cout->end();++it) {
        std::string pair_tag = it->first;
        size_t pair_count = it->second;
        std::string first_state = pair_tag.substr(0,pair_tag.find(SPERATOR_FLAG,0));
        std::string second_state = pair_tag.substr(pair_tag.find(SPERATOR_FLAG,0)+1);
        size_t tag_count = ptr_cout->find(first_state)->second;
        double trans_prob = (double) pair_count / tag_count;
        ptr_prob->insert(make_pair(pair_tag,trans_prob));
    }
    for (std::map<std::string, double >::iterator iit = ptr_state_to_x_prob_map_->begin();iit!=ptr_state_to_x_prob_map_->end();++iit) {
       // std::cout << "The count of "<<iit->first<<" is "<<iit->second<<std::endl;
    }
}

void DatasetMgr::Calc() {
    GenerateStateTransitionVector();
    GenerateCountMap(ptr_pair_tag_vector_,ptr_pair_tag_count_map_, true);
    GenerateCountMap(ptr_tag_vector_,ptr_tag_count_map_, false);
    CalcProb(ptr_tag_count_map_,ptr_pair_tag_count_map_,ptr_state_to_state_prob_map_);
    GenerateCountMap(ptr_x_tag_vector_,ptr_x_tag_count_map_, true);
    CalcProb(ptr_tag_count_map_,ptr_x_tag_count_map_,ptr_state_to_x_prob_map_);
}

std::map<std::string, double> *DatasetMgr::GetStateTransProbMap() const {
    return ptr_state_to_state_prob_map_;
}

std::map<std::string, double> *DatasetMgr::GetEmissionProbMap() const {
    return ptr_state_to_x_prob_map_;
}

std::vector<std::string> *DatasetMgr::GetTestFeatureVector() const {
    return ptr_test_x_vector_;
}

std::vector<std::string> *DatasetMgr::GetTestFlagVector() const {
    return ptr_test_tag_vector_;
}

std::set<std::string> *DatasetMgr::GetTagSet() const {
    return ptr_tag_set_;
}

std::vector<std::string> *DatasetMgr::GetTrainingXVector() const {
    return ptr_x_vector_;
}

std::set<std::string> *DatasetMgr::GetTrainingXSet() const {
    return ptr_x_set_;
}

size_t DatasetMgr::GetNumOfTrainingSeqs() const {
    return num_of_training_setence_;
}


