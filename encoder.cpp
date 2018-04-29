//
// Created by  ngs on 27/04/2018.
//

#include "encoder.h"

Encoder::Encoder(const char *training_file):train_file_(training_file) {
    ptr_datamgr_ = new DatasetMgr();
    ptr_datamgr_->OpenDataSet(training_file, true);
}

int Encoder::StartTraining() {
    ptr_datamgr_->Calc();
}

DatasetMgr* Encoder::GetPtrDatamgr_() const {
    return ptr_datamgr_;
}