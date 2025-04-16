cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(metalama, CXX)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(kernel)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/lib)

find_library(FOUNDATION Foundation REQUIRED)
find_library(METAL Metal REQUIRED)
find_library(METALFX MetalFX REQUIRED)
find_library(QUARTZCORE QuartzCore REQUIRED)

find_package(simdjson CONFIG REQUIRED)


#aux_source_directory(src metalama_SOURCES)
set(metalama_METALLIB metalama.metallib)
set(metalama_SOURCES src/main.cc)

add_executable(metalama_main ${metalama_SOURCES})
add_dependencies(metalama_main ${metalama_METALLIB})

target_link_libraries(metalama_main ${FOUNDATION})
target_link_libraries(metalama_main ${METAL})
target_link_libraries(metalama_main ${METALFX})
target_link_libraries(metalama_main ${QUARTZCORE})


set(metalama_TENSOR_SOURCES src/safetensor.cc)
add_executable(metalama_parse ${metalama_TENSOR_SOURCES})
target_link_libraries(metalama_parse simdjson::simdjson)
