cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(metalchat, CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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


set(metalchat_METALLIB metalchat.metallib)
aux_source_directory(src metalchat_SOURCES)


add_library(metalchat SHARED ${metalchat_SOURCES})
add_dependencies(metalchat ${metalchat_METALLIB})
set_target_properties(metalchat PROPERTIES CXX_STANDARD 23)
set_target_properties(metalchat PROPERTIES VERSION ${metalchat_LIBRARY_VERSION})
set_target_properties(metalchat PROPERTIES SOVERSION ${metalchat_LIBRARY_SOVERSION})
target_link_libraries(metalchat simdjson::simdjson)
target_link_libraries(metalchat cppcodec::cppcodec)
target_link_libraries(metalchat metal-cpp::metal-cpp)
target_link_libraries(metalchat PCRE2::8BIT)


# Install the library files.
install(DIRECTORY include/metalchat DESTINATION metalchat)
#install(FILES include/metalchat.h DESTINATION include)
install(TARGETS metalchat DESTINATION lib)
