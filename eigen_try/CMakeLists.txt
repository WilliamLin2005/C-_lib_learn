cmake_minimum_required(VERSION 3.0)
project(EigenTest)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(eigen_try1 src/eigen_try1.cpp)
add_executable(eigen_try2 src/eigen_try2.cpp)