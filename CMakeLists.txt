cmake_minimum_required(VERSION 3.17)
project(tiny_CNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
#find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("Found OpenMP")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS})
endif()
add_executable(tiny_CNN main.cpp src/layer.h src/utils.h src/layer/relu.cpp src/layer/relu.h src/layer/max_pooling.cpp src/layer/max_pooling.h src/layer/conv.h src/layer/conv.cpp src/network.h src/network.cpp
        src/mnist.h src/mnist.cpp src/layer/conv_relu_max_pooling.h src/layer/conv_relu_max_pooling.cpp src/loss.h src/loss/mockloss.h src/loss/mockloss.cpp)

