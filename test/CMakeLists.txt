include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_STANDARD 23)

find_package(Catch2 3 REQUIRED)

file(GLOB metalchat_TEST_SOURCES *.cc)


foreach(metalchat_TEST_SOURCE ${metalchat_TEST_SOURCES})
    # Extract extension from the file name.
    get_filename_component(metalchat_TEST ${metalchat_TEST_SOURCE} NAME_WE)

    add_executable(${metalchat_TEST} ${metalchat_TEST_SOURCE})
    target_link_libraries(${metalchat_TEST} Catch2::Catch2WithMain)
    target_link_libraries(${metalchat_TEST} MetalChat)

    target_compile_definitions(
        ${metalchat_TEST}
        PUBLIC __lib_metalchat_test_fixture_directory="${PROJECT_SOURCE_DIR}/test_fixture"
    )

    # Define a new test from the source file.
    add_test(
        NAME ${metalchat_TEST}
        COMMAND ${metalchat_TEST} --skip-benchmarks
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )
endforeach(metalchat_TEST_SOURCE)
