//
// Created by  ngs on 27/04/2018.
//

#ifndef CLIONTEST_ENCODER_H
#define CLIONTEST_ENCODER_H

#include "datasetmgr.h"

class Encoder{
public:
    explicit Encoder(const char* training_file);
    void StartTraining();
    DatasetMgr * GetPtrDatamgr() const;
private:
    const std::string train_file_;
    DatasetMgr *ptr_datamgr_;
};

#endif //CLIONTEST_ENCODER_H
