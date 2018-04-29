#include <iostream>
#include <fstream>
#include "vector"
#include "datasetmgr.h"
#include "encoder.h"
#include "decoder.h"

int main(int argc, char **argv) {
    Encoder *ptr_encoder = new Encoder(argv[1]);
    ptr_encoder->StartTraining();
    Decoder *ptr_decoder = new Decoder(argv[2], ptr_encoder->GetPtrDatamgr_());
    ptr_decoder->Decoding();
}