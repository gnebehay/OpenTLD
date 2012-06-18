cmake_minimum_required(VERSION 2.6)

add_library(tld
    Clustering.cpp
    DetectionResult.cpp
    DetectorCascade.cpp
    EnsembleClassifier.cpp
    ForegroundDetector.cpp
    MedianFlowTracker.cpp
    NNClassifier.cpp
    TLD.cpp
    TLDUtil.cpp
    VarianceFilter.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})