cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(metalchat, CXX)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(kernel)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/lib)

find_library(FOUNDATION Foundation REQUIRED)
find_library(METAL Metal REQUIRED)
find_library(METALFX MetalFX REQUIRED)
find_library(QUARTZCORE QuartzCore REQUIRED)

find_package(simdjson CONFIG REQUIRED)


#aux_source_directory(src metalchat_SOURCES)
set(metalchat_METALLIB metalchat.metallib)
set(metalchat_SOURCES src/main.cc)

add_executable(metalchat_main ${metalchat_SOURCES})
add_dependencies(metalchat_main ${metalchat_METALLIB})

target_link_libraries(metalchat_main simdjson::simdjson)
target_link_libraries(metalchat_main ${FOUNDATION})
target_link_libraries(metalchat_main ${METAL})
target_link_libraries(metalchat_main ${METALFX})
target_link_libraries(metalchat_main ${QUARTZCORE})
