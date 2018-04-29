cmake_minimum_required(VERSION 3.10)
project(cliontest)

set(CMAKE_CXX_STANDARD 11)

add_executable(cliontest main.cpp datasetmgr.h encoder.h encoder.cpp datasetmgr.cpp common.h decoder.h decoder.cpp)