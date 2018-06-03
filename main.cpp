#include <iostream>
#include <fstream>
#include "vector"
#include "datasetmgr.h"
#include "encoder.h"
#include "decoder.h"
#include "unsupervised.h"
#include "em.h"
#include "SoftEM.h"
int main(int argc, char **argv) {
    Encoder *ptr_encoder = new Encoder(argv[1]);
    ptr_encoder->StartTraining();
 //   Decoder *ptr_decoder = new Decoder(argv[2], ptr_encoder->GetPtrDatamgr());
 //   ptr_decoder->Decoding();
 //   Learning *ptr_unpserpvised = new Learning(argv[1],ptr_encoder->GetPtrDatamgr());
  //  ptr_unpserpvised->StartTraining();
    //EM *ptr_em = new EM(argv[1],ptr_encoder->GetPtrDatamgr());
    SoftEM *ptr_em = new SoftEM(argv[1],ptr_encoder->GetPtrDatamgr());
    ptr_em->Learning();
}