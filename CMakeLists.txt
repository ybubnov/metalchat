cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

project(metalchat LANGUAGES CXX Swift)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_FRAMEWORK TRUE)

set(metalchat_LIBRARY_VERSION "1.0.0")
set(metalchat_LIBRARY_SOVERSION "1")

enable_testing()


add_subdirectory(kernel)
add_subdirectory(test)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(cppcodec CONFIG REQUIRED)
find_package(simdjson CONFIG REQUIRED)
find_package(metal-cpp CONFIG REQUIRED)
find_package(PCRE2 CONFIG REQUIRED)


set(metalchat_FRAMEWORK MetalChat)
set(metalchat_METALLIB metalchat.metallib)
aux_source_directory(src metalchat_SOURCES)

# Metal shaders are pre-compiled and therefore only available after building an
# appropriate target. Since those shaders should be a part of the framework, we
# need to include them as a resource. But resources could be only existing files.
# This trick (marking file as generated) fixes this issue.
set_source_files_properties(${metalchat_METALLIB} PROPERTIES GENERATED TRUE)

add_library(${metalchat_FRAMEWORK} SHARED ${metalchat_SOURCES})
add_dependencies(${metalchat_FRAMEWORK} ${metalchat_METALLIB})
target_sources(${metalchat_FRAMEWORK} PRIVATE ${metalchat_METALLIB})

set_target_properties(${metalchat_FRAMEWORK} PROPERTIES
    CXX_STANDARD 23
    FRAMEWORK TRUE
    FRAMEWORK_VERSION A
    RESOURCE ${metalchat_METALLIB}
    MACOSX_FRAMEWORK_IDENTIFIER com.cmake.metalchat
    MACOSX_FRAMEWORK_BUNDLE_NAME ${metalchat_FRAMEWORK}
    MACOSX_FRAMEWORK_BUNDLE_VERSION ${metalchat_LIBRARY_SOVERSION}
    MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${metalchat_LIBRARY_VERSION}
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "MetalChat Developer"
)

target_link_libraries(${metalchat_FRAMEWORK}
    simdjson::simdjson
    cppcodec::cppcodec
    metal-cpp::metal-cpp
    PCRE2::8BIT
)
