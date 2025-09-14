cmake_minimum_required(VERSION 3.12)
project(helios)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(threads_test threads.cpp)
