//
// Created by  ngs on 27/04/2018.
//

#ifndef CLIONTEST_ENCODER_H
#define CLIONTEST_ENCODER_H

#include "datasetmgr.h"

class Encoder{
public:
    explicit Encoder(const char* training_file);
    int StartTraining();
    DatasetMgr * GetPtrDatamgr_() const;
private:
    const std::string train_file_;
    DatasetMgr *ptr_datamgr_;
};

#endif //CLIONTEST_ENCODER_H
