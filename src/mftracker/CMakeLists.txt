cmake_minimum_required(VERSION 2.6)

add_library(mftracker
    bb.cpp
    bb_predict.cpp
    fbtrack.cpp
    lk.cpp
    median.cpp)
