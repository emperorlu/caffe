cmake_minimum_required(VERSION 2.6)
project(deal_model)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(CAFFE_ROOT /home/kevin/caffe)
INCLUDE_DIRECTORIES(/opt/OpenBLAS/include)
INCLUDE_DIRECTORIES(${CAFFE_ROOT}/include)
INCLUDE_DIRECTORIES(${CAFFE_ROOT}/build/src)
INCLUDE_DIRECTORIES(${CAFFE_ROOT}/build/include)
#FIND_LIBRARY(RUNTIME_LIB rt ${CAFFE_ROOT}/build/lib )
#FIND_LIBRARY(BLAS_LIB bl  /opt/OpenBLAS/lib NO_DEFAULT_PATH)
#FIND_LIBRARY(COMMON cm /usr/local/lib NO_DEFAULT_PATH)
link_directories(${CAFFE_ROOT}/build/lib /opt/OpenBLAS/lib /lib /usr/local/lib)

link_libraries(${RUNTIME_LIB}  )
set(SOURCE_FILES
        deal_model.cpp
        )
add_executable(deal_model.bin ${SOURCE_FILES})
target_link_libraries(deal_model.bin -lcaffe -lopenblas -lglog -lgflags
        -lprotobuf -lboost_system -lboost_filesystem)




