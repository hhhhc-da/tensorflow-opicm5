cmake_minimum_required(VERSION 3.10)
project(MNIST)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/include/opencv4)
include_directories(include)
include_directories(include/src)

link_directories(lib)

add_executable(main mnist.cc)

target_link_libraries(main opencv_core opencv_imgcodecs opencv_highgui tensorflow_cc libtensorflow_framework.so.2.17.0)

