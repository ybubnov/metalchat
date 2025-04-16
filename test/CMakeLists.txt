include_directories(${PROJECT_SOURCE_DIR}/include)
#include_directories(${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_CXX_STANDARD 23)
#add_definitions(-std=c++2b)

#find_library(FOUNDATION Foundation REQUIRED)
#find_library(METAL Metal REQUIRED)
#find_library(QUARTZCORE QuartzCore REQUIRED)
find_package(metal-cpp CONFIG REQUIRED)
find_package(simdjson CONFIG REQUIRED)
find_package(Catch2 3 REQUIRED)

file(GLOB metalchat_TEST_SOURCES *.cc)


foreach(metalchat_TEST_SOURCE ${metalchat_TEST_SOURCES})
    # Extract extension from the file name.
    get_filename_component(metalchat_TEST ${metalchat_TEST_SOURCE} NAME_WE)

    add_executable(${metalchat_TEST} ${metalchat_TEST_SOURCE})
    #target_link_libraries(${metalchat_TEST} metal-cpp::metal-cpp)
    #target_link_libraries(${metalchat_TEST} ${FOUNDATION})
    #target_link_libraries(${metalchat_TEST} ${METAL})
    #target_link_libraries(${metalchat_TEST} ${QUARTZCORE})
    #target_link_libraries(${metalchat_TEST} simdjson::simdjson)
    target_link_libraries(${metalchat_TEST} Catch2::Catch2WithMain)
    target_link_libraries(${metalchat_TEST} metalchat)

    # Define a new test from the source file.
    add_test(
        NAME ${metalchat_TEST}
        COMMAND ${metalchat_TEST}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )
endforeach(metalchat_TEST_SOURCE)
