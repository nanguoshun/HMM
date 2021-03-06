//
// Created by  ngs on 27/04/2018.
//

#include "encoder.h"

Encoder::Encoder(const char *training_file):train_file_(training_file) {
    ptr_datamgr_ = new DatasetMgr(true);
    ptr_datamgr_->OpenDataSet(training_file, true);
}

void Encoder::StartTraining() {
    ptr_datamgr_->Calc();
}

DatasetMgr* Encoder::GetPtrDatamgr() const {
    return ptr_datamgr_;
}