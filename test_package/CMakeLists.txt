cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

project(test_package LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)

find_package(metalchat CONFIG REQUIRED)

add_executable(${PROJECT_NAME} main.cc)
target_link_libraries(${PROJECT_NAME} PRIVATE metalchat::metalchat)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)
