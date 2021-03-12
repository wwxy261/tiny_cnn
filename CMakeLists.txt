cmake_minimum_required(VERSION 3.17)
project(tiny_CNN)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MyTarget PUBLIC OpenMP::OpenMP_CXX)
endif()
add_executable(tiny_CNN main.cpp src/layer.h src/utils.h src/layer/relu.cpp src/layer/relu.h src/layer/max_pooling.cpp src/layer/max_pooling.h src/layer/conv.h src/layer/conv.cpp src/network.h src/network.cpp
        src/mnist.h src/mnist.cpp src/layer/conv_relu_max_pooling.h src/layer/conv_relu_max_pooling.cpp)

