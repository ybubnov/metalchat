cmake_minimum_required(VERSION 3.31 FATAL_ERROR)
file(STRINGS version.txt metalchat_LIBRARY_VERSION)

project(metalchat
        VERSION ${metalchat_LIBRARY_VERSION}
        DESCRIPTION "Llama inference for Apple Devices"
        HOMEPAGE_URL "https://github.com/ybubnov/metalchat"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_FRAMEWORK TRUE)
set(metalchat_LIBRARY_SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR})

add_subdirectory(kernel)

# The variable BUILD_TESTING is set by conan generator through, when a build
# option build.tools:skip_test is set to 'True'. In this case, unit tests won't
# be compiled and executed.
if (NOT BUILD_TESTING STREQUAL OFF)
    enable_testing()
    add_subdirectory(test)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(cppcodec CONFIG REQUIRED)
find_package(jsoncons CONFIG REQUIRED)
find_package(mbits-mstch CONFIG REQUIRED)
find_package(metal-cpp CONFIG REQUIRED)
find_package(rapidhash CONFIG REQUIRED)
find_package(PCRE2 CONFIG REQUIRED)
find_package(replxx)


set(metalchat_FRAMEWORK MetalChat)
set(metalchat_METALLIB metalchat.metallib)
set(metalchat_MODULEMAP ${CMAKE_CURRENT_SOURCE_DIR}/module.modulemap)

aux_source_directory(src metalchat_SOURCES)
aux_source_directory(program metalchat_PROGRAM)
aux_source_directory(include metalchat_HEADERS)

# Metal shaders are pre-compiled and therefore only available after building an
# appropriate target. Since those shaders should be a part of the framework, we
# need to include them as a resource. But resources could be only existing files.
# This trick (marking file as generated) fixes this issue.
set_source_files_properties(${metalchat_METALLIB} PROPERTIES GENERATED TRUE)

add_library(${metalchat_FRAMEWORK} SHARED ${metalchat_SOURCES} ${metalchat_HEADERS})
add_dependencies(${metalchat_FRAMEWORK} ${metalchat_METALLIB})
target_sources(${metalchat_FRAMEWORK} PRIVATE ${metalchat_METALLIB})

set_target_properties(${metalchat_FRAMEWORK} PROPERTIES
    CXX_STANDARD 23
    # LINK_FLAGS_RELEASE "-Wl,-dead_strip -Wl,-x"
    FRAMEWORK TRUE
    FRAMEWORK_VERSION A
    RESOURCE ${metalchat_METALLIB}
    MACOSX_FRAMEWORK_IDENTIFIER com.cmake.metalchat
    MACOSX_FRAMEWORK_BUNDLE_NAME ${metalchat_FRAMEWORK}
    MACOSX_FRAMEWORK_BUNDLE_VERSION ${metalchat_LIBRARY_SOVERSION}
    MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${metalchat_LIBRARY_VERSION}
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "MetalChat Developer"
)

set(metalchat_FRAMEWORK_DIR "$<TARGET_FILE_DIR:${metalchat_FRAMEWORK}>")
set(metalchat_FRAMEWORK_ROOT "${metalchat_FRAMEWORK_DIR}/../..")
set(metalchat_FRAMEWORK_MODULES ${metalchat_FRAMEWORK_DIR}/Modules)
set(metalchat_FRAMEWORK_HEADERS ${metalchat_FRAMEWORK_DIR}/Headers)

# Swift framework require module map to correctly link C++ symbols, so put this
# module map file manually to the target framework and link Modules directory, so
# that clang linker could find it.
add_custom_target(copy_modulemap ALL
    DEPENDS ${metalchat_MODULEMAP}
    COMMAND ${CMAKE_COMMAND} -E
        make_directory ${metalchat_FRAMEWORK_MODULES}
    COMMAND ${CMAKE_COMMAND} -E
        copy ${metalchat_MODULEMAP} ${metalchat_FRAMEWORK_MODULES}
    COMMAND ${CMAKE_COMMAND} -E
        create_symlink Versions/Current/Modules ${metalchat_FRAMEWORK_ROOT}/Modules
    COMMENT "Copying ${metalchat_MODULEMAP} into the OS X framework ${metalchat_FRAMEWORK_MODULES}"
)

add_custom_target(copy_headers ALL
    COMMAND ${CMAKE_COMMAND} -E
        make_directory ${metalchat_FRAMEWORK_HEADERS}
    COMMAND ${CMAKE_COMMAND} -E
        copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include/metalchat ${metalchat_FRAMEWORK_HEADERS}
    COMMAND ${CMAKE_COMMAND} -E
        create_symlink Versions/Current/Headers ${metalchat_FRAMEWORK_ROOT}/Headers
    COMMENT "Copying headers into the OS X framework ${metalchat_FRAMEWORK_HEADERS}"
)

target_link_libraries(${metalchat_FRAMEWORK}
    cppcodec::cppcodec
    jsoncons
    rapidhash::rapidhash
    mbits::mstch
    metal-cpp::metal-cpp
    PCRE2::8BIT
)


if (replxx_FOUND)
    add_executable(${CMAKE_PROJECT_NAME} ${metalchat_PROGRAM})
    target_link_libraries(${CMAKE_PROJECT_NAME}
        ${metalchat_FRAMEWORK}
        replxx::replxx
    )
endif()
