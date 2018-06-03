cmake_minimum_required(VERSION 3.10)
project(HMM)

set(CMAKE_CXX_STANDARD 11)

add_executable(HMM main.cpp datasetmgr.h encoder.h encoder.cpp datasetmgr.cpp common.h decoder.h decoder.cpp unsupervised.h unsupervised.cpp em.h em.cpp fwbw.h fwbw.cpp)