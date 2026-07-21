cmake_minimum_required(VERSION 3.20)
project(perennis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PERENNIS_ENABLE_ASAN "Build with AddressSanitizer" OFF)
option(PERENNIS_ENABLE_UBSAN "Build with UndefinedBehaviorSanitizer" OFF)
option(PERENNIS_BUILD_TESTS "Build tests" ON)
option(PERENNIS_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

add_library(perennis_warnings INTERFACE)
target_compile_options(perennis_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
    $<$<BOOL:${PERENNIS_WARNINGS_AS_ERRORS}>:-Werror>
)

add_library(perennis_sanitizers INTERFACE)
if(PERENNIS_ENABLE_ASAN)
    target_compile_options(perennis_sanitizers INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(perennis_sanitizers INTERFACE -fsanitize=address)
endif()
if(PERENNIS_ENABLE_UBSAN)
    target_compile_options(perennis_sanitizers INTERFACE -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_options(perennis_sanitizers INTERFACE -fsanitize=undefined)
endif()

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml)

add_library(perennis INTERFACE)
target_include_directories(perennis INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(perennis INTERFACE perennis_warnings perennis_sanitizers ggml)

if(PERENNIS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
